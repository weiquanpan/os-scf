import unittest,tempfile
from .common import *
from .evaluate import rounded_samples

class ProtocolTests(unittest.TestCase):
    def test_unknown_does_not_expand_training_vocabulary(self):
        codec=Codec(pd.DataFrame({'a':['known','known']}),['a'])
        before=codec.names; encoded=codec.encode(pd.DataFrame({'a':['heldout']}))
        self.assertEqual(codec.names,before);self.assertEqual(codec.names[0][encoded[0,0]],'__UNKNOWN__')
    def test_fixed_schema_domain(self):
        codec=Codec(pd.DataFrame({'a':[0,0]}),['a'],{'a':['0','1']})
        self.assertEqual(codec.encode(pd.DataFrame({'a':[1]}))[0,0],1)
        with self.assertRaises(ValueError):codec.encode(pd.DataFrame({'a':[2]}))
    def test_evaluation_does_not_advance_training_rng(self):
        seed(42);expected=torch.rand(3,device=DEVICE);seed(42)
        with isolated_rng(71):torch.rand(300,device=DEVICE);np.random.rand(20)
        self.assertTrue(torch.equal(expected,torch.rand(3,device=DEVICE)))
    def test_sliced_wasserstein_identity(self):
        a=np.random.default_rng(1).normal(size=(128,4));self.assertEqual(sliced_wasserstein(a,a,seed=17),0)
    def test_taxi_conditioned_reconstruction(self):
        codec=Codec(pd.DataFrame({c:['1','2'] for c in TAXI_CATS}),TAXI_CATS)
        x=torch.ones(20,len(TAXI_COLS),device=DEVICE)*10;spec=TaxiSpec(x,codec);spec.lower.zero_();spec.upper.fill_(1000)
        b=NS(name='taxi',codec=codec,spec=spec,free=list(range(len(TAXI_FREE))))
        u=torch.ones(2,len(TAXI_FREE),device=DEVICE);k=torch.zeros(2,len(TAXI_CATS),device=DEVICE,dtype=torch.long);k[1,0]=1
        result=decode(b,u,k);self.assertEqual(float(result[0,-1]),6.);self.assertEqual(float(result[1,-1]),8.)

if __name__=='__main__':unittest.main()
