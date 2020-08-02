
import { WebCNN } from './Webcnn.js'
import { LAYER_TYPE_INPUT_IMAGE  } from './defines.js';
const totalTrainingImages = 50000;
const examplesPerImageFile = 10000;
let trainingSetImages = new Array( 5 );
let trainingSetCTX = new Array( 5 );
let trainingSetImagesCounter = 0;
let trainingSetImagesLoaded = [ false, false, false, false, false ];
let allTrainingSetImagesLoaded = false;

let validationSetImage;
let validationSetImageLoaded = false;
let validationSetCTX;

let miniBatchSize = 20;

let ctx_error;

let epochsPerPixel;

let cnn;

function Init()
{
	let canvas = document.getElementById( "weightsCanvas" );
	ctx_error = canvas.getContext( "2d" );
	ctx_error.strokeStyle = "red";
    console.log('ctx_error', ctx_error);
	epochsPerPixel = totalTrainingImages / ( miniBatchSize * ctx_error.canvas.width );
    console.log(epochsPerPixel);
	// Load training data
	for ( let i = 0; i < 5; ++i )
	{
		trainingSetImages[ i ] = new Image();
		trainingSetImages[ i ].src = './images/mnist_training_' + i + '.png';
		trainingSetImages[ i ].onload = OnTrainingSetImageLoaded;
	}
	
	validationSetImage = new Image();
	validationSetImage.src = '../images/mnist_validation.png';
	validationSetImage.onload = OnValidationSetImageLoaded;

    createDefaultNetwork();
}

function OnTrainingSetImageLoaded( e )
{
	var i = 0;
	for ( i = 0; i < 5; ++i )
	{
        console.log('trainingSetImages[ i ] ', trainingSetImages[ i ] );
        console.log('this', this );
		if ( trainingSetImages[ i ] == this )
		{
            console.log("y");
			let canvas = document.createElement( "canvas" );
			canvas.width = 2800;
			canvas.height = 2800;
			trainingSetCTX[ i ] = canvas.getContext( "2d" );
			trainingSetCTX[ i ].drawImage( trainingSetImages[ i ], 0, 0 );
			trainingSetImagesLoaded[ i ] = true;
			var d = document.createElement('div');
			d.innerText = `trainingSetImages ${i}`;
			document.getElementById("app").appendChild(d);
            document.getElementById("app").appendChild(canvas);
			break;
		}
	}

	trainingSetImagesCounter++;
	if ( trainingSetImagesCounter == trainingSetImages.length )
	{
		allTrainingSetImagesLoaded = true;
        console.log( "All training Images Loaded" );
        console.log(trainingSetCTX[0]);
        
	}
	checkImagesLoaded();
}

function OnValidationSetImageLoaded( e )
{
	let canvas = document.createElement( "canvas" );
	canvas.width = 2800;
	canvas.height = 2800;
	validationSetCTX = canvas.getContext( "2d" );
	validationSetCTX.drawImage( validationSetImage, 0, 0 );
	validationSetImageLoaded = true;
	console.log( "Validation Images Loaded" );

	checkImagesLoaded();
}

function checkImagesLoaded()
{
	if ( trainingSetImagesLoaded[ 0 ] && validationSetImageLoaded )
	{
		// const btn = document.getElementById( "startButton" );
		// btn.innerHTML = "Start Training";
		console.log('You can start training');
	}
}

function createDefaultNetwork()
{
	cnn = new WebCNN( miniBatchSize );
	cnn.newLayer( { name: "image", type: LAYER_TYPE_INPUT_IMAGE, width: 24, height: 24, depth: 1 } );
	// cnn.newLayer( { name: "conv1", type: LAYER_TYPE_CONV, units: 10, kernelWidth: 5, kernelHeight: 5, strideX: 1, strideY: 1, padding: false } );
	// cnn.newLayer( { name: "pool1", type: LAYER_TYPE_MAX_POOL, poolWidth: 2, poolHeight: 2, strideX: 2, strideY: 2 } );
	// cnn.newLayer( { name: "conv2", type: LAYER_TYPE_CONV, units: 20, kernelWidth: 5, kernelHeight: 5, strideX: 1, strideY: 1, padding: false } );
	// cnn.newLayer( { name: "pool2", type: LAYER_TYPE_MAX_POOL, poolWidth: 2, poolHeight: 2, strideX: 2, strideY: 2 } );
	// cnn.newLayer( { name: "out", type: LAYER_TYPE_FULLY_CONNECTED, units: 10, activation: ACTIVATION_SOFTMAX } );
	cnn.initialize();

	cnn.setLearningRate( 0.01 );
	cnn.setMomentum( 0.9 );
	cnn.setLambda( 0.0 );

	console.log(cnn);
}

Init();