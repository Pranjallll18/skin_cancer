const imageUpload = document.getElementById("imageUpload");
const predictBtn = document.getElementById("predictBtn");
const preview = document.getElementById("preview");
const previewBox = document.getElementById("previewBox");
const resultBox = document.getElementById("result");

imageUpload.addEventListener("change", () => {
  const file = imageUpload.files[0];
  if (file) {
    previewBox.style.display = "block";
    preview.src = URL.createObjectURL(file);
  }
});

predictBtn.addEventListener("click", async () => {
  const file = imageUpload.files[0];
  if (!file) {
    alert("Please upload an image first.");
    return;
  }

  const formData = new FormData();
  formData.append("image", file);

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    resultBox.innerText = `Prediction: ${data.prediction}`;
  } catch (error) {
    resultBox.innerText = "Prediction failed. Please try again.";
    console.error(error);
  }
});
