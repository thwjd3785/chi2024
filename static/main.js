// Capture the webcam image
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        video.srcObject = "stream";
    });
document.getElementById('capture-btn').addEventListener('click', () => {
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    // Crop the image using Cropper.js
    const cropper = new Cropper(canvas, {
        aspectRatio: 1 / 1,
        viewMode: 1,
        dragMode: 'move',
        cropBoxResizable: true,
        cropBoxMovable: true,
        minCropBoxWidth: 100,
        minCropBoxHeight: 100,
        ready() {
            cropper.setCropBoxData({
                width: 200,
                height: 200,
            });
        },
    });
    // Save the cropped image to the img element
    document.getElementById('image').src = cropper.getCroppedCanvas().toDataURL();
});
// Save the cropped image to the server
document.getElementById('save-btn').addEventListener('click', () => {
    const dataUrl = document.getElementById('image').src;
    $.ajax({
        type: "POST",
        /*url: "/save_image/",
        data: {
            image: dataUrl,
        },*/
        success: function () {
            alert('Image saved successfully!');
        },
        error: function () {
            alert('Failed to save image!');
        },
    });
});