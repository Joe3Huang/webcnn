import { ACTIVATION_LINEAR } from './defines.js';
/*
	Abstract network layer base class
 */
export class Layer
{
	/*
	 name: String, the name of the layer used for debugging and logging
	 units: Number of neurons/nodes/units in a FC layer, or kernels in a conv layer
	 inputs: Number of incoming connections, required for determining variance in Xavier style weight initialization
	 */
	constructor( name, units )
	{
		this.name = name;
		this.units = units;

		// Defaults
        this.activation = ACTIVATION_LINEAR;
        this.network = null;
    }
    
    setNetwork(theCNN) {
        this.network = theCNN;
    }

	isLastLayer() { return this.nextLayer == undefined; }

	setInputLayer( inputLayer )
	{
		this.prevLayer = inputLayer;
		this.inputDimensions = inputLayer.outputDimensions;
	}

	setOutputLayer( outputLayer )
	{
		this.nextLayer = outputLayer;
	}

	commitMiniBatch()
	{
		// No-op in layers that don't override this (layers without weights and biases)
	}
}
