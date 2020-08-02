/*
 A simple class for storing and comparing dimensions of input volumes, output volumes, kernels, etc.
 */
export class Dimensions
{
	constructor( width, height, depth )
	{
		// Flooring all values to guarantee integer dimensions.
		this.width = Math.floor( width );
		this.height = Math.floor( height );
		this.depth = Math.floor( depth );
	}

	getSize()
	{
		return this.width * this.height * this.depth;
	}

	// For debugging sanity checks
	static equal( dim1, dim2 )
	{
		return ( dim1.width == dim2.width && dim1.height == dim2.height && dim1.depth == dim2.depth );
	}
}