var model;
async function load_model() {
    model = await tf.loadGraphModel('TFJS/model.json');
}

function predictImage() {
   // console.log('Processing Image..');

    //load image
    let image = cv.imread(canvas);

    //convert the imaage to B/W
    cv.cvtColor(image, image, cv.COLOR_RGBA2GRAY, 0);
    cv.threshold(image, image, 175, 255, cv.THRESH_BINARY)

    //Find the contours
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();
    cv.findContours(image, contours, hierarchy,
        cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);

    //Calculate bounding rectangle
    let cnt = contours.get(0);
    let rect = cv.boundingRect(cnt);

    // //crop the image
    image = image.roi(rect);

    //calculate new size
    var height = image.rows;
    var width = image.cols;
   // console.log('Before ' + width + " : " + height);
    if (height > width) {
        height = 20;
        const scale_factor = image.rows / height;
        width = Math.round(width / scale_factor);
    }
    else {
        width = 20;
        const scale_factor = image.cols / width;
        height = Math.round(height / scale_factor);
    }

    //Resize image
    let newSize = new cv.Size(width, height);
    cv.resize(image, image, newSize, 0, 0, cv.INTER_AREA)

   // console.log('New :' + image.cols + ' : ' + image.rows);
    //add padding
    const LEFT = Math.ceil(4 + (20 - width) / 2);
    const RIGHT = Math.floor(4 + (20 - width) / 2);
    const TOP = Math.ceil(4 + (20 - height) / 2);
    const BOTTOM = Math.floor(4 + (20 - height) / 2);

    const BLACK = new cv.Scalar(0, 0, 0, 0);
    cv.copyMakeBorder(image, image, TOP, BOTTOM, LEFT, RIGHT, cv.BORDER_CONSTANT, BLACK);//top, bottom, left,right
   // console.log('copyBorder : ' + image.cols + ' : ' + image.rows);

    //find Center of Mass
    cv.findContours(image, contours, hierarchy,cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);
    cnt = contours.get(0);
    const Moments = cv.moments(cnt, false);
    const cx = Moments.m10 / Moments.m00;
    const cy = Moments.m01 / Moments.m00;
    //console.log(`M00: ${Moments.m00} M01: ${Moments.m01} M10: ${Moments.m10}`)
    //console.log('COM '+cx+' : '+cy)

    //shift the image
    const X_SHIFT = Math.round(image.cols/2 - cx);
    const Y_SHIFT = Math.round(image.rows/2 - cy);
    newSize = new cv.Size(image.cols, image.rows);
    const M = cv.matFromArray(2, 3, cv.CV_64FC1, [1, 0, X_SHIFT, 0, 1, Y_SHIFT]);
    cv.warpAffine(image, image, M, newSize, cv.INTER_LINEAR, cv.BORDER_CONSTANT, BLACK);
    
    //normalize the pixel values
                    //(image.data contains pixel values in integer from 0 to 255(inclusive)
                        //Float32Array converts image.data(array of integer to array of decimals)
    let pixelValues = Float32Array.from(image.data).map((num) => { 
        return num / 255.0;
    });
   // console.log('scaled vlaues '+pixelValues);
    
    //create a tensor
    X = tf.tensor([pixelValues]);
    // console.log(`Shape of tensor ${X.shape}`);
    // console.log(`dtype of Tensot ${X.dtype}`);

    //predict
    const result = model.predict(X);
    result.print();

   // console.log(tf.memory());

    const output = result.dataSync()[0];


    //Tesitng Only 
    // const outputCanvas = document.createElement('CANVAS');
    // cv.imshow(outputCanvas, image);
    // document.body.appendChild(outputCanvas);

    //Cleanup 
    image.delete();
    contours.delete();
    cnt.delete();
    hierarchy.delete();
    M.delete();

    X.dispose();
    result.dispose();

    return output;

}