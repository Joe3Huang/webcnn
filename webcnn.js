import {
    FORWARD_MODE_TRAINING, 
    BACKWARD_MODE_TRAINING, 
    LAYER_TYPE_INPUT_IMAGE,
    LAYER_TYPE_CONV

} from './defines.js';

import { InputImageLayer } from './InputImageLayer.js';

/*
	This class is the network wrapper which owns hyperparameters and
	provides the input/output API to the Layers of the network composed within it
 */
export class WebCNN
{
	constructor( miniBatchSize )
	{
		this.layers = [];
		this.nextLayerIndex = 0;

		// Defaults to training mode
		this.forwardMode = FORWARD_MODE_TRAINING;
		this.backwardMode = BACKWARD_MODE_TRAINING;

		// Default hyperparameters
		this.learningRate = 0.01;
		this.momentum = 0.9;

		// L2 Regularization
		this.lambda =  0.0;

		// For auto-detection of solved MLP networks
		this.solutionEpoch = 0;
		this.trainingError = 0;

		// For benchmarking performance
		this.forwardTime = 0;
		this.backwardTime = 0;

		this.miniBatchSize = miniBatchSize;
	}

	// Public API getters and setters
	setMomentum( momentum ) { this.momentum = momentum; }
	getMomentum() { return this.momentum; }

	setLearningRate( rate ) { this.learningRate = rate; }
	getLearningRate() { return this.learningRate; }

	setLambda( lambda ) { this.lambda = lambda; }
	getLambda() { return this.lambda; }

	// Create a layer from a descriptor object and add it to the layer stack
	// intention: public API
	newLayer( layerDesc )
	{
		console.log( "Creating layer "+this.nextLayerIndex+": "+layerDesc.name );

		let newLayer;

		switch( layerDesc.type )
		{
			case LAYER_TYPE_INPUT_IMAGE:
			{
				newLayer = new InputImageLayer( layerDesc.name, layerDesc.width, layerDesc.height, layerDesc.depth );
				break;
			}

			// case LAYER_TYPE_CONV:
			// {
			// 	newLayer = new ConvLayer(	layerDesc.name, layerDesc.units, layerDesc.kernelWidth, layerDesc.kernelHeight,
			// 								layerDesc.strideX, layerDesc.strideY, layerDesc.padding );
			// 	break;
			// }

			// case LAYER_TYPE_MAX_POOL:
			// {
			// 	newLayer = new MaxPoolLayer( layerDesc.name, layerDesc.poolWidth, layerDesc.poolHeight,
			// 								 layerDesc.strideX, layerDesc.strideY );
			// 	break;
			// }

			// case LAYER_TYPE_FULLY_CONNECTED:
			// {
			// 	newLayer = new FCLayer( layerDesc.name, layerDesc.units, layerDesc.activation );
			// 	break;
			// }
		}

		newLayer.network = this;

		if ( this.nextLayerIndex == 0 )
		{
			if ( newLayer.type != LAYER_TYPE_INPUT_IMAGE && newLayer.type != LAYER_TYPE_INPUT )
			{
				throw "First layer must be input type";
			}
		}
		else
		{
			// Link with previous layer
			const prevLayer = this.layers[ this.nextLayerIndex - 1 ];
			prevLayer.setOutputLayer(newLayer);
			newLayer.setInputLayer(prevLayer);
		}

		newLayer.layerIndex = this.nextLayerIndex;
		this.layers[ this.nextLayerIndex ] = newLayer;
		this.nextLayerIndex++;
	}

	// intention: public API
	initialize()
	{
		// Placeholder. Right now, the last-added layer is implicitly the output, but it's
		// my intention to formalize "finalizing" and initializing some parameters of the
		// network in this function.
	}

	// intention: public API
	trainCNNClassifier( imageDataArray, imageLabelsArray )
	{
		this.miniBatchSize = Math.floor( Math.min( imageDataArray.length, imageLabelsArray.length ) );

		this.batchLearningRate = this.learningRate / this.miniBatchSize;

		this.forwardMode = FORWARD_MODE_TRAINING;
		this.backwardMode = BACKWARD_MODE_TRAINING;
		this.trainingError = 0;

		this._cnnSetTrainingTargets( imageLabelsArray );

		let t0 = Date.now();
		this._cnnForward( imageDataArray );
		let t1 = Date.now();
		this._cnnBackward();
		this._endCNNMiniBatch();

		this.trainingError /= this.miniBatchSize;
		this.forwardTime = ( t1 - t0 ) / this.miniBatchSize;
		this.backwardTime = ( Date.now() - t1) / this.miniBatchSize;
	}

	// intention: private
	_cnnSetTrainingTargets( imageLabelsArray )
	{
		this.targetValues = new Array( this.miniBatchSize );
		for ( var example = 0; example < this.miniBatchSize; ++example )
		{
			this.targetValues[ example ] = new Array( this.layers[ this.layers.length - 1 ].units );
			for ( var unit = 0; unit < this.layers[ this.layers.length - 1 ].units; ++unit )
			{
				this.targetValues[ example ][ unit ] = ( unit == imageLabelsArray[ example ] ) ? 1 : 0;
			}
		}
	}

	// intention: private
	_cnnForward( imageDataArray )
	{
		this.layers[ 0 ].forward( imageDataArray );
		for ( let i = 1; i < this.layers.length; ++i )
		{
			this.layers[ i ].forward();
		}
	}

	// intention: private
	_cnnBackward()
	{
		for ( let i = this.layers.length - 1; i > 0; --i )
		{
			this.layers[ i ].backward();
		}
	}

	// intention: private
	_endCNNMiniBatch()
	{
		for ( let i = this.layers.length - 1; i > 0; --i )
		{
			this.layers[ i ].commitMiniBatch();
		}
	}

	// intention: public API
	classifyImages( imageDataArray )
	{
		let outputLayer = this.layers[ this.layers.length - 1 ];
		this.miniBatchSize = imageDataArray.length;
		this.batchLearningRate = this.learningRate;

		this.forwardMode = FORWARD_MODE_EVALUATE;

		this.layers[ 0 ].forward( imageDataArray );
		for ( var i = 1; i < this.layers.length; ++i )
		{
			this.layers[ i ].forward();
		}

		this.forwardMode = FORWARD_MODE_TRAINING;
		return outputLayer.output;
	}
}