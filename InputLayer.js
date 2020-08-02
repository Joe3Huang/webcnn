/*
 	Input values layer, just some numbers. This layer option is for making small
 	non-convolutional networks for testing and debugging purposes.
 */
import { LAYER_TYPE_INPUT, ACTIVATION_NONE } from './defines.js';
export class InputLayer extends Layer
{
	constructor( name, units )
	{
		super( name, units, 0, units );
        this.type = LAYER_TYPE_INPUT;
		this.activation = ACTIVATION_NONE;
        this.outputDimensions = new Dimensions( 1, 1, units );
	}

	forward()
	{
		for ( var example = 0; example < this.network.miniBatchSize; ++example )
		{
			this.output[ example ] = new Array3D( this.outputDimensions, INIT_ZEROS );
			for ( let unit = 0; unit < this.units; ++unit )
			{
				this.output[ example ].setValue( 0, 0, unit, this.network.inputValues[ example ][ unit ] );
			}
		}
	}
}