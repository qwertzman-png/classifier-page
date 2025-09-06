document.addEventListener('DOMContentLoaded', () => {
    const imageSelector = document.getElementById('image-selector');
    const selectedImage = document.getElementById('selected-image');
    const predictionResult = document.getElementById('prediction-result');

    let model;

    // Load the model
    async function loadModel() {
        try {
            // The path is relative to the index.html file
            const modelUrl = './model_js/model.json';
            model = await tf.loadGraphModel(modelUrl);
            predictionResult.innerText = 'Model loaded. Please select an image.';
        } catch (error) {
            console.error('Error loading model:', error);
            predictionResult.innerText = 'Error loading model.';
        }
    }

    loadModel();

    imageSelector.addEventListener('change', e => {
        const file = e.target.files[0];
        if (!file) {
            return;
        }

        const reader = new FileReader();
        reader.onload = () => {
            selectedImage.src = reader.result;
            selectedImage.style.display = 'block';
            // Add a delay to ensure the image is rendered before prediction
            setTimeout(() => predict(selectedImage), 100);
        };
        reader.readAsDataURL(file);
    });

    async function predict(imgElement) {
        if (!model) {
            predictionResult.innerText = 'Model not loaded yet.';
            return;
        }

        predictionResult.innerText = 'Predicting...';

        try {
            // Pre-process the image:
            // 1. Convert the image to a tensor
            let tensor = tf.browser.fromPixels(imgElement)
                // 2. Resize it to the expected input size of the model
                // NOTE: You might need to change this to the actual size your model expects.
                // Common sizes are 224x224, 299x299, etc.
                .resizeNearestNeighbor([224, 224])
                // 3. Normalize the pixel values to be between 0 and 1
                .toFloat()
                .div(tf.scalar(255.0))
                // 4. Add a batch dimension
                .expandDims();

            // Run the prediction
            const predictions = await model.predict(tensor).data();
            
            // Post-process the output
            // This assumes a classification model. You may need to adjust this
            // based on what your model actually outputs.
            const topPrediction = predictions[0]; // Example: get the first value
            
            // Display the result
            // You'll want to map this value to a meaningful label.
            predictionResult.innerText = `Prediction Score: ${topPrediction.toFixed(4)}`;

        } catch (error) {
            console.error('Error during prediction:', error);
            predictionResult.innerText = 'Error during prediction.';
        }
    }
});
