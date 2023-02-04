use crate::tensors::{Input, Kernel, Output};
use algebra::{fp_64::Fp64Parameters, FixedPoint, FixedPointParameters, FpParameters, PrimeField};
use crypto_primitives::AdditiveShare;
use num_traits::Zero;
use std::{
    marker::PhantomData,
    ops::{Add, AddAssign, Mul},
};
use tch::nn;

#[derive(Debug)]
pub struct FullyConnectedParamsD<F, C> {
    pub weights: Kernel<C>,
    pub bias: Kernel<C>,
    pub tch_config: Option<nn::Linear>,
    pub eval_method: crate::EvalMethod,
    _variable: PhantomData<F>,
}

unsafe impl<F, C> Send for FullyConnectedParamsD<F, C> {}
unsafe impl<F, C> Sync for FullyConnectedParamsD<F, C> {}

impl<F, C> FullyConnectedParamsD<F, C>
where
    F: Zero + Mul<C, Output = F> + AddAssign + Add<Output = F> + Copy,
    C: Copy + Into<F>,
{
    // size of weights: (1,1,output_size,input_size)
    pub fn new(weights: Kernel<C>, bias: Kernel<C>) -> Self {
        let kernel_dims = weights.dim();
        let bias_dims = bias.dim();
        assert!(
            (bias_dims.0 == kernel_dims.0)
                && (bias_dims.1 == kernel_dims.1)
                && (bias_dims.2 == kernel_dims.2)
                && (bias_dims.3 == kernel_dims.3)
        );
        Self {
            weights,
            bias,
            tch_config: None,
            eval_method: crate::EvalMethod::Naive,
            _variable: PhantomData,
        }
    }

    pub fn calculate_output_size(
        &self,
        (batch_size, _, msl, d): (usize, usize, usize, usize),
    ) -> (usize, usize, usize, usize) {
	let output_size = self.weights.dim().2;
	let insize=self.weights.dim().3;
        assert_eq!(d, insize);

        (batch_size, 1, msl, output_size)
    }

    pub fn fully_connected_naive(&self, input: &Input<F>, out: &mut Output<F>) {
        let (batch_size, _, msl, d) = input.dim();
        let (_,_,outputsize, insize) = self.weights.dim();
        let (o_batch_size, _, ..) = out.dim();
        assert!(
            (o_batch_size == batch_size),
            "Shape doesn't match: input: {:?}, weights: {:?}, output: {:?}",
            input.dim(),
            self.weights.dim(),
            out.dim()
        );

        let i_zero = ndarray::Axis(0);
	out.axis_iter_mut(i_zero)
	    .zip(input.axis_iter(i_zero))
	    .for_each(|(mut outS3,inpS3)|{
		outS3.axis_iter_mut(i_zero)
		    .zip(inpS3.axis_iter(i_zero))
		    .for_each(|(mut outs2, inps2)|{
			outs2=self.weights.dot(inps2.t()).t()+self.bias[s![0,0,..,..]];
		    })

	    });

    }
}

impl<P: FixedPointParameters> FullyConnectedParamsD<AdditiveShare<FixedPoint<P>>, FixedPoint<P>>
where
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
{
    pub fn new_with_gpu(
        vs: &nn::Path,
        weights: Kernel<FixedPoint<P>>,
        bias: Kernel<FixedPoint<P>>,
    ) -> Self {
        let (out_channels, in_channels, ..) = weights.dim();
        let device = vs.device();
        let weights_tensor = weights.to_tensor().to_device(device);
        let bias_tensor = bias
            .to_tensor()
            .reshape(&[bias.dim().0 as i64])
            .to_device(device);
        let mut out = Self::new(weights, bias);
        out.eval_method = crate::EvalMethod::TorchDevice(device);

        let mut tch_config = nn::linear(
            vs,
            in_channels as i64,
            out_channels as i64,
            Default::default(),
        );
        tch_config.ws = weights_tensor;
        tch_config.bs = bias_tensor;
        out.tch_config = Some(tch_config);
        out
    }
}
