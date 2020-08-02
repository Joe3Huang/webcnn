
export const LAYER_TYPE_CONV =  'convLayer'; 
export const LAYER_TYPE_MAX_POOL =  'maxPoolLayer';
export const LAYER_TYPE_INPUT_IMAGE =  'inputImageLayer';
export const LAYER_TYPE_INPUT =  'inputLayer';
export const LAYER_TYPE_FULLY_CONNECTED =  'FCLayer';
export const ACTIVATION_RELU =  'relu';
export const ACTIVATION_TANH =  'tanh';
export const ACTIVATION_LINEAR =  'linear';
export const ACTIVATION_SOFTMAX=  'softmax';
export const INIT_ZEROS =  'initZeros';
export const INIT_XAVIER =  'xavier';
export const INIT_GLOROT_UNIFORM =  'glorot_uniform';

export const FORWARD_MODE_TRAINING =  'training';		// Normal training mode
export const FORWARD_MODE_EVALUATE =  'evaluate';		// Normal evaluation mode where no values need to be calculated for backprop
export const BACKWARD_MODE_TRAINING =  'training';
export const BACKWARD_MODE_DREAMING =  'dreaming';