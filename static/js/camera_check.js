// Function to check camera access
async function checkCamera() {
    const cameraCheckElement = document.getElementById("camera-check");
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        cameraCheckElement.textContent = "Camera is working!";
        cameraCheckElement.classList.remove("text-warning");
        cameraCheckElement.classList.add("text-success");
        stream.getTracks().forEach(track => track.stop()); // Stop the camera stream
        return true;
    } catch (error) {
        cameraCheckElement.textContent = "Camera not detected";
        cameraCheckElement.classList.remove("text-warning");
        cameraCheckElement.classList.add("text-danger");
        return false;
    }
}

// Function to check microphone access
async function checkMicrophone() {
    const micCheckElement = document.getElementById("mic-check");
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        micCheckElement.textContent = "Microphone is working!";
        micCheckElement.classList.remove("text-warning");
        micCheckElement.classList.add("text-success");
        stream.getTracks().forEach(track => track.stop()); // Stop the mic stream
        return true;
    } catch (error) {
        micCheckElement.textContent = "Microphone not detected";
        micCheckElement.classList.remove("text-warning");
        micCheckElement.classList.add("text-danger");
        return false;
    }
}

// Main function to check both camera and microphone
async function checkDevices() {
    const cameraWorking = await checkCamera();
    const micWorking = await checkMicrophone();

    const proceedButton = document.getElementById("proceed-btn");
    if (cameraWorking && micWorking) {
        proceedButton.style.display = "block";
        proceedButton.onclick = () => {
            window.location.href = "/interview";
        };
    } else {
        alert("Please ensure both camera and microphone are working!");
    }
}

// Start the device check when the page loads
window.onload = checkDevices;
