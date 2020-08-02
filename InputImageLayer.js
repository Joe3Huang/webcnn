/*
 	Input image layer
 */
import { LAYER_TYPE_INPUT_IMAGE } from './defines.js';
import { Layer } from './Layer.js';
import { Dimensions } from './Dimensions.js';
export class InputImageLayer extends Layer
{
	constructor( layerName, imageWidth, imageHeight, imageDepth )
	{
		if ( imageDepth != 1 && imageDepth != 3 )
		{
			// Only 1 or 3 components per pixel are supported (i.e. greyscale or 3-component color)
			throw "Invalid input image depth, must be 1 (greyscale) or 3 (RGB)";
		}

		super( layerName, 1 );
		this.type = LAYER_TYPE_INPUT_IMAGE;
		this.outputDimensions = new Dimensions( imageWidth, imageHeight, imageDepth );
		this.output = [];
	}

	// This layer always just sets output values on forward propagation
	// imageDataArray must be an Array of ImageData objects
	forward( imageDataArray )
	{
		for ( var example = 0; example < this.network.miniBatchSize; ++example )
		{
			this.output[ example ] = new Array3D( this.outputDimensions, INIT_ZEROS );
			this.output[ example ].setFromImageData( imageDataArray[ example ], this.outputDimensions.depth );
		}
	}

	// In normal classification/regression, back propagation does not alter the input image,
	// this function is only for deep dreaming style image reconstruction.
	backward()
	{
		// Reserved for future functionality
	}
}