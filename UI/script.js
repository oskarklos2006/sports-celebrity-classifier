const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");

/* MODEL LABELS — MUST MATCH TRAINING */
const PLAYER_LABELS = {
    "Courtois": 0,
    "Cristiano Ronaldo": 1,
    "Dybala": 2,
    "Kross": 3,
    "Lionel Messi": 4,
    "Mohamed Salah": 5,
    "Neymar": 6,
    "Pogba": 7
};

const uploadArea = document.querySelector(".upload-area");

imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (!file) return;

    uploadArea.classList.add("active");
    uploadArea.querySelector(".upload-text").innerText = file.name;

    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";
});

uploadArea.addEventListener("dragover", e => {
    e.preventDefault();
    uploadArea.classList.add("active");
});

uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("active");
});

uploadArea.addEventListener("drop", e => {
    e.preventDefault();
    uploadArea.classList.add("active");

    const file = e.dataTransfer.files[0];
    imageInput.files = e.dataTransfer.files;

    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";
});


async function predict() {
    const file = imageInput.files[0];
    if (!file) {
        return;
    }

    const formData = new FormData();
    formData.append("image", file);

    const response = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    const resultWrapper = document.getElementById("resultWrapper");
    const errorMessage = document.getElementById("errorMessage");
    const confidenceSection = document.getElementById("confidenceSection");
    const playerEl = document.getElementById("player");
    const confidenceEl = document.getElementById("confidence");

    // Always show result container
    resultWrapper.classList.remove("hidden");

    // RESET UI
    errorMessage.classList.add("hidden");
    confidenceSection.classList.remove("hidden");
    playerEl.innerText = "";
    confidenceEl.innerText = "";

    // HANDLE ERROR FROM BACKEND
    if (!response.ok || data.error || !data.player) {
        errorMessage.innerText =
            data.error || "No clear face detected. Please upload a clear face image.";

        errorMessage.classList.remove("hidden");
        confidenceSection.classList.add("hidden");
        return;
    }

    // SUCCESS
    const labelNumber = PLAYER_LABELS[data.player];

    playerEl.innerText = `Prediction: ${data.player} (${labelNumber})`;
    confidenceEl.innerText = `Confidence: ${Math.round(data.confidence * 100)}%`;

    const container = document.getElementById("confidenceBars");
    container.innerHTML = "";

    const sorted = Object.entries(data.all_confidences)
        .sort((a, b) => b[1] - a[1]);

    for (const [player, prob] of sorted) {
        const percent = Math.round(prob * 100);

        const row = document.createElement("div");
        row.className = "confidence-row";

        row.innerHTML = `
            <div class="confidence-label">
                <span>${player} (${PLAYER_LABELS[player]})</span>
                <span>${percent}%</span>
            </div>
            <div class="confidence-bar-bg">
                <div class="confidence-bar" style="width:${percent}%"></div>
            </div>
        `;

        container.appendChild(row);
    }
}


