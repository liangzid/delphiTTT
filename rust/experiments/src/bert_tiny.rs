use ::neural_network as nn;
use nn::{
    layers::{convolution::Padding, Layer},
    NeuralNetwork,
};
use rand::{CryptoRng, RngCore};

use super::*;

// initialize a pre-computed bert-tiny model.
pub fn construct_berttiny<R:RngCore + CryptoRng>(
    vs:Option<&tch::nn::Path>,
    batch_size:usize,
    rng:&mut R,
)->NeuralNetwork<TenBitAS,TenBitExpFP>{

    let mut network = match &vs{
	Some(vs) => NeuralNetwork{
	    layers:vec![],
	    eval_method: ::neural_network::EvalMethod::TorchDevice(vs.device()),
	},
	None => NeuralNetwork{
	    layers: vec![],
	    ..Default::defualt()
	},
    };

    // Dimensions of input text sequence representation.
    let input_dims=(batch_size,1,128,768);

    let inter_dim=3072;
    let cls_dim=2;

    // initial feature transformation 
    let fc_indims=(1,1,1,input_dims.3);
    let (initial_d_fc, _)=sample_fc_layer(vs,fc_indims,
					  inter_dim,rng);
    network.layers.push(Layer::LL(initial_d_fc));

    // initial attention transformation
    let (initial_m_fc, _)=sample_fc_layer(vs,input_dims[2],
					  input_dims[2],rng);
    network.layers.push(Layer::LL(initial_m_fc));
    add_activation_layer(&mut network, &relu_layers);

    //  intermediate feature transformation
    let (inter_d_fc, _)=sample_fc_layer(vs,inter_dims,
					  inter_dims,rng);
    network.layers.push(Layer::LL(inter_d_fc));

    //  intermediate attention transformation
    let (inter_m_fc, _)=sample_fc_layer(vs,input_dims[2],
					  input_dims[2],rng);
    network.layers.push(Layer::LL(inter_m_fc));
    add_activation_layer(&mut network, &relu_layers);

    //  final feature transformation
    let (final_d_fc, _)=sample_fc_layer(vs,inter_dim,
					  cls_dim,rng);
    network.layers.push(Layer::LL(final_d_fc));

    assert!(network.validate());

    network
}
