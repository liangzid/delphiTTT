use ::neural_network as nn;
use nn::{
    layers::{convolution::Padding, Layer},
    NeuralNetwork,
};
use rand::{CryptoRng, RngCore};

use super::*;

pub fn construct_minionn<R: RngCore + CryptoRng>(
    vs: Option<&tch::nn::Path>,
    batch_size: usize,
    // num_poly: usize,
    rng: &mut R,
) -> NeuralNetwork<TenBitAS, TenBitExpFP> {
    let relu_layers=vec![1,2,3,4,5,6,7];

    let mut network = match &vs {
        Some(vs) => NeuralNetwork {
            layers: vec![],
            eval_method: ::neural_network::EvalMethod::TorchDevice(vs.device()),
        },
        None => NeuralNetwork {
            layers: vec![],
            ..Default::default()
        },
    };

    // Dimensions of input tensor.
    let msl=128;
    let d=768;
    let input_dims = (batch_size, msl, d, 0);

    // set BERT layers

    // 1
    let kernel_dims = (64, 3, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    // 10
    let fc_input_dims = network.layers.last().unwrap().output_dimensions();
    let (fc, _) = sample_fc_layer(vs, fc_input_dims, 10, rng);
    network.layers.push(Layer::LL(fc));
    assert!(network.validate());

    network
}
