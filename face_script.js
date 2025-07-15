const video = document.getElementById("video");

navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
  video.srcObject = stream;
});

function captureImage() {
  const canvas = document.createElement("canvas");
  canvas.width = 320;
  canvas.height = 240;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, 320, 240);

  const base64Image = canvas.toDataURL("image/jpeg");

  document.getElementById("status").innerText = "⏳ Analyzing...";

  fetch("https://neurotrack-backend-eauk.onrender.com/predict-face", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: base64Image })
  })
  .then(res => res.json())
  .then(data => {
    document.getElementById("result").innerText = "🧠 Detected Emotion: " + data.emotion;
    document.getElementById("status").innerText = "✅ Done!";
  })
  .catch(err => {
    document.getElementById("result").innerText = "❌ Error detecting emotion.";
    console.error(err);
  });
}
