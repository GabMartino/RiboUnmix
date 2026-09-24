from __future__ import annotations
import copy
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pack_padded_sequence

from Tests.test_alpha_causality_modes import _small_model
from Models.RiboUnmixLightningModule import NegativeBinomialProfileLoss
from Utils.reference_campaign import (task_matrix,permute_reference_weights,check_execution_gate,
    derived_common_reference,shared_extreme_mask,matched_random_masks,object_hash)


def small_model(policy='equal',unity=False):
    torch.manual_seed(42)
    model=_small_model('learned')
    model.mean_correction='unity' if unity else 'learned'
    model.selected_dataset_names=('d0','d1');model.selected_dataset_ids=(0,1)
    ref=dict(weighting=policy,quality_rank_power=1.,chunk_size=1)
    if policy=='explicit':ref['explicit_weights']={'d0':.25,'d1':.75}
    model._configure_gamma_centering({'gamma_centering':{'mode':'fixed_reference','reference':ref}},
        reference_dataset_names=['d0','d1'],reference_dataset_ids=[0,1],reference_dataset_quality_weights=[1.,.5])
    return model


def batch():
    rng=torch.Generator().manual_seed(91)
    x=torch.randn(2,8,5,generator=rng)
    return dict(x_packed=pack_padded_sequence(x,torch.tensor([8,8]),batch_first=True),
        mask=torch.ones(2,8,dtype=torch.bool),codon_ids=torch.randint(0,8,(2,8),generator=rng),
        id_datasets=torch.tensor([0,1]),target=torch.randint(1,9,(2,8),generator=rng).float(),
        sample_ids=['t0','t1'],transcript_group_index=torch.tensor([0,1]))


class CampaignTests(unittest.TestCase):
    def test_matrix_is_48_or_96_not_a_hidden_n_series(self):
        for campaign,n in [('main',48),('extended',96)]:
            t=task_matrix(campaign)
            self.assertEqual(len(t),n);self.assertEqual(t.task_id.nunique(),n)
            self.assertEqual(t.milestone.eq(1).sum(),8)
            self.assertEqual(len(t[t.arm=='shared_only']),12)
        t=task_matrix('main');self.assertEqual(set(t[t.arm.str.startswith('shuffled')].training_seed),{42})
        self.assertEqual(len(task_matrix('main',[42],1)),16)

    def test_shuffles_preserve_dataset_not_source_concentration(self):
        table=pd.DataFrame([dict(panel_id=f'panel_{p:02}',dataset_id=f'p{p}d{i}',dataset_order=i,
            source_family=f'p{p}f{i//3}',global_rank=1+i+29*(p-1),raw_reference_score_q=(115-i-29*(p-1))/115)
            for p in range(1,5) for i in range(29 if p<=2 else 28)])
        a,diag=permute_reference_weights(table);b,_=permute_reference_weights(table)
        pd.testing.assert_frame_equal(a,b)
        self.assertEqual(len(a),342)
        for panel,g in table.groupby('panel_id'):
            pi=g.raw_reference_score_q.to_numpy();pi=pi/pi.sum()
            for _,rows in a[a.panel_id==panel].groupby('permutation_id'):
                np.testing.assert_array_equal(np.sort(rows.pi),np.sort(pi))
                self.assertEqual(set(rows.dataset_id),set(g.dataset_id))
        self.assertTrue(diag.groupby('panel_id').maximum_source_mass.nunique().gt(1).any())

    def test_failed_support_cannot_be_overridden_by_authorization(self):
        m=dict(prerequisite_failures=['validation support failed'],planned_training_count=48,plan_hash='x',candidate_partition_sha256='y')
        with self.assertRaisesRegex(ValueError,'prerequisites'):
            check_execution_gate(m,approved_plan_hash='x',approved_partition_manifest=Path('unused'),max_new_trainings=48,authorize_training=True)
        m['prerequisite_failures']=[]
        with self.assertRaisesRegex(ValueError,'authorize-training'):
            check_execution_gate(m,approved_plan_hash='x',approved_partition_manifest=None,max_new_trainings=8,authorize_training=False)
        with self.assertRaisesRegex(ValueError,'task cap'):
            check_execution_gate(m,approved_plan_hash='x',approved_partition_manifest=None,max_new_trainings=None,authorize_training=True)
        self.assertNotEqual(object_hash({'seed':42}),object_hash({'seed':43}))

    def test_initial_parameters_match_across_policies_and_shared_arm(self):
        original=dict(small_model().named_parameters())
        for policy,unity in [('quality_rank',False),('explicit',False),('equal',True)]:
            for name,p in small_model(policy,unity).named_parameters():
                torch.testing.assert_close(p,original[name],rtol=0,atol=0)

    def test_unity_mean_alpha_head_and_context_gradient_paths(self):
        model=small_model(unity=True);data=batch()
        mu,log_alpha,extra=model(**data)
        torch.testing.assert_close(extra['gamma'],torch.ones_like(extra['gamma']),rtol=0,atol=0)
        torch.testing.assert_close(mu,extra['scale_dt'][:,None]*extra['L_bio'])
        loss=NegativeBinomialProfileLoss(experiment_mode='standard_nb',nb_mean_gradient_beta=0)(
            mu_phys=None,log_mu_phys=extra['log_mu'],log_sigma=log_alpha,y_true=data['target'],mask=data['mask'])
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.biological_model.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.dataset_bias_model.log_sigma_head.parameters()))
        self.assertTrue(all(p.grad is None for p in model.dataset_bias_model.local_context_gru.parameters()))
        self.assertIsNone(model.dataset_bias_model.dataset_embedding.weight.grad)
        self.assertTrue(all(p.grad is None for p in model.dataset_bias_model.observation_bias_head.parameters()))
        state=copy.deepcopy(model.state_dict());other=small_model();other.load_state_dict(state)
        self.assertEqual(other.mean_correction,'unity')

    def test_explicit_reference_checkpoint_and_chunk_gradient_equivalence(self):
        a=small_model('explicit').eval();b=small_model('explicit').eval();b.gamma_reference_chunk_size=2
        data=batch();x=a(**data);y=b(**data)
        torch.testing.assert_close(x[0],y[0],rtol=1e-5,atol=1e-6)
        for output,model in [(x,a),(y,b)]:output[0].square().sum().backward()
        for pa,pb in zip(a.parameters(),b.parameters()):
            if pa.grad is not None: torch.testing.assert_close(pa.grad,pb.grad,rtol=1e-4,atol=1e-5)
        c=small_model();c.load_state_dict(copy.deepcopy(a.state_dict()))
        self.assertEqual(c.gamma_centering_weighting,'explicit')
        torch.testing.assert_close(c.gamma_reference_weights,torch.tensor([.25,.75]))

    def test_common_reference_equal_identity_full_cds_normalization(self):
        rng=np.random.default_rng(61);L=np.exp(rng.normal(size=80));L/=L.mean()
        g=rng.normal(size=(4,80));g-=g.mean(axis=0)
        out=derived_common_reference(np.log(L),g,np.ones(80,bool))
        np.testing.assert_allclose(out['values'],L,rtol=1e-13)
        self.assertAlmostEqual(out['Z'],1.)
        interior=out['values'][20:-20]
        self.assertFalse(np.isclose(interior.mean(),1.))
        shifted=derived_common_reference(np.log(L)+1000,g,np.ones(80,bool))
        np.testing.assert_allclose(shifted['values'],L,rtol=1e-12)
        self.assertIsNone(shifted['Z']);self.assertTrue(np.isfinite(shifted['log_Z']))
        with self.assertRaisesRegex(ValueError,'aligned'):derived_common_reference(np.log(L),g[:,:50],np.ones(80,bool))

    def test_shared_peak_union_and_matched_random_masks(self):
        x=np.ones((8,100));x[np.arange(8),np.arange(8)]=100
        removed=shared_extreme_mask(x,np.ones(100,bool),.01)
        self.assertEqual(removed.sum(),8)  # Union exceeds nominal 1%.
        masks=list(matched_random_masks(np.ones(100,bool),removed.sum(),91,100))
        self.assertEqual(len(masks),100);self.assertTrue(all(m.sum()==8 for m in masks))
        np.testing.assert_array_equal(masks[0],next(matched_random_masks(np.ones(100,bool),8,91)))


if __name__=='__main__':unittest.main()
