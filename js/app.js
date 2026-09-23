/* =====================================================
   FIELDLENS FRONTEND
   API SERVER
===================================================== */

const API_BASE_URL =
    "https://naturals-make-evaluate-optical.trycloudflare.com";

const PREDICT_URL =
    `${API_BASE_URL}/predict`;



/* =====================================================
   PAGE ELEMENTS
===================================================== */

const homePage =
    document.getElementById("homePage");

const analysePage =
    document.getElementById("analysePage");



/* =====================================================
   NAVIGATION
===================================================== */

const startAnalysis =
    document.getElementById("startAnalysis");

const homeButton =
    document.getElementById("homeButton");

const analyseHomeButton =
    document.getElementById("analyseHomeButton");

const backButton =
    document.getElementById("backButton");



function showAnalysePage() {

    if (!homePage || !analysePage) {
        return;
    }

    homePage.classList.add("hidden");

    analysePage.classList.remove("hidden");

    window.scrollTo(0, 0);

}


function showHomePage() {

    if (!homePage || !analysePage) {
        return;
    }

    analysePage.classList.add("hidden");

    homePage.classList.remove("hidden");

    window.scrollTo(0, 0);

}


if (startAnalysis) {

    startAnalysis.addEventListener(
        "click",
        showAnalysePage
    );

}


if (homeButton) {

    homeButton.addEventListener(
        "click",
        showHomePage
    );

}


if (analyseHomeButton) {

    analyseHomeButton.addEventListener(
        "click",
        showHomePage
    );

}


if (backButton) {

    backButton.addEventListener(
        "click",
        showHomePage
    );

}



/* =====================================================
   ANALYSIS PAGE
===================================================== */

const fileInput =
    document.getElementById("fileInput");

const uploadButton =
    document.getElementById("uploadButton");

const cameraButton =
    document.getElementById("cameraButton");

const dropZone =
    document.getElementById("dropZone");

const preview =
    document.getElementById("preview");

const previewImage =
    document.getElementById("previewImage");

const fileName =
    document.getElementById("fileName");

const removeButton =
    document.getElementById("removeButton");

const analyseButton =
    document.getElementById("analyseButton");

const statusBox =
    document.getElementById("status");



let selectedFile = null;

let previewURL = null;



/* =====================================================
   FILE SELECT
===================================================== */

if (uploadButton) {

    uploadButton.addEventListener(
        "click",
        () => {

            fileInput.removeAttribute(
                "capture"
            );

            fileInput.click();

        }
    );

}


if (cameraButton) {

    cameraButton.addEventListener(
        "click",
        () => {

            fileInput.setAttribute(
                "capture",
                "environment"
            );

            fileInput.click();

        }
    );

}


if (fileInput) {

    fileInput.addEventListener(
        "change",
        () => {

            const file =
                fileInput.files[0];

            if (file) {

                validateFile(file);

            }

        }
    );

}



/* =====================================================
   DRAG AND DROP
===================================================== */

if (dropZone) {


    [
        "dragenter",
        "dragover"
    ].forEach(
        eventName => {

            dropZone.addEventListener(
                eventName,
                event => {

                    event.preventDefault();

                    dropZone.classList.add(
                        "dragging"
                    );

                }
            );

        }
    );


    [
        "dragleave",
        "drop"
    ].forEach(
        eventName => {

            dropZone.addEventListener(
                eventName,
                event => {

                    event.preventDefault();

                    dropZone.classList.remove(
                        "dragging"
                    );

                }
            );

        }
    );


    dropZone.addEventListener(
        "drop",
        event => {

            const file =
                event.dataTransfer.files[0];

            if (file) {

                validateFile(file);

            }

        }
    );

}



/* =====================================================
   VALIDATE IMAGE
===================================================== */

function validateFile(file) {

    clearStatus();


    if (
        !file.type ||
        !file.type.startsWith("image/")
    ) {

        showError(
            "Please select a valid image file."
        );

        return;

    }


    if (
        file.size >
        10 * 1024 * 1024
    ) {

        showError(
            "Image is too large. Maximum size is 10 MB."
        );

        return;

    }


    selectedFile =
        file;


    if (previewURL) {

        URL.revokeObjectURL(
            previewURL
        );

    }


    previewURL =
        URL.createObjectURL(
            file
        );


    previewImage.src =
        previewURL;


    fileName.textContent =
        file.name;


    preview.style.display =
        "block";


    analyseButton.disabled =
        false;

}



/* =====================================================
   REMOVE IMAGE
===================================================== */

if (removeButton) {

    removeButton.addEventListener(
        "click",
        () => {

            selectedFile = null;

            fileInput.value = "";

            preview.style.display =
                "none";

            analyseButton.disabled =
                true;

            clearStatus();

        }
    );

}



/* =====================================================
   ANALYZE
===================================================== */

if (analyseButton) {

    analyseButton.addEventListener(
        "click",
        analyzeImage
    );

}


async function analyzeImage() {


    if (!selectedFile) {

        showError(
            "Please select an image first."
        );

        return;

    }


    analyseButton.disabled =
        true;


    analyseButton.textContent =
        "Analyzing...";


    showLoading(
        "Uploading image and analyzing the crop..."
    );


    try {


        /* -----------------------------------------
           CREATE MULTIPART FORM
        ----------------------------------------- */

        const formData =
            new FormData();


        /*
         * IMPORTANT:
         *
         * Your FastAPI endpoint expects:
         *
         * file
         *
         * So don't change this name.
         */

        formData.append(
            "file",
            selectedFile
        );



        /* -----------------------------------------
           SEND TO GLOBAL API
        ----------------------------------------- */

        const response =
            await fetch(
                PREDICT_URL,
                {
                    method: "POST",
                    body: formData
                }
            );



        /* -----------------------------------------
           READ RESPONSE
        ----------------------------------------- */

        let result;


        try {

            result =
                await response.json();

        }

        catch {

            throw new Error(
                "The API returned an invalid response."
            );

        }



        /* -----------------------------------------
           API ERROR
        ----------------------------------------- */

        if (!response.ok) {

            throw new Error(
                result.error ||
                result.detail ||
                "The API rejected the image."
            );

        }



        /* -----------------------------------------
           LOW CONFIDENCE / RETAKE
        ----------------------------------------- */

        if (
            result.status ===
            "retake"
        ) {


            showError(
                result.message ||
                "The image is not clear enough. Please take another photo."
            );


            analyseButton.disabled =
                false;


            analyseButton.textContent =
                "Analyze leaf →";


            return;

        }



        /* -----------------------------------------
           STORE API RESULT
        ----------------------------------------- */

        sessionStorage.setItem(
            "predictionResult",
            JSON.stringify(result)
        );



        /*
         * Store image as a data URL.
         *
         * This is better than storing the
         * temporary blob URL because result.html
         * can be opened separately.
         */

        const imageData =
            await fileToDataURL(
                selectedFile
            );


        sessionStorage.setItem(
            "uploadedImage",
            imageData
        );



        /* -----------------------------------------
           GO TO RESULT PAGE
        ----------------------------------------- */

        window.location.href =
            "./result.html";


    }

    catch (error) {


        console.error(
            "FieldLens API error:",
            error
        );


        /*
         * Browser CORS / network errors usually
         * appear here as "Failed to fetch".
         */

        if (
            error.message ===
            "Failed to fetch"
        ) {

            showError(
                "Could not connect to the API server. Check that the API is online and allows requests from this website."
            );

        }

        else {

            showError(
                error.message ||
                "Something went wrong while analyzing the image."
            );

        }


        analyseButton.disabled =
            false;


        analyseButton.textContent =
            "Analyze leaf →";

    }

}



/* =====================================================
   FILE → DATA URL
===================================================== */

function fileToDataURL(file) {

    return new Promise(
        (
            resolve,
            reject
        ) => {


            const reader =
                new FileReader();


            reader.onload =
                () => {

                    resolve(
                        reader.result
                    );

                };


            reader.onerror =
                reject;


            reader.readAsDataURL(
                file
            );

        }
    );

}



/* =====================================================
   STATUS
===================================================== */

function showLoading(message) {

    if (!statusBox) {
        return;
    }


    statusBox.className =
        "status loading";


    statusBox.textContent =
        message;

}


function showError(message) {

    if (!statusBox) {
        return;
    }


    statusBox.className =
        "status error";


    statusBox.textContent =
        message;

}


function clearStatus() {

    if (!statusBox) {
        return;
    }


    statusBox.className =
        "status";


    statusBox.textContent =
        "";

}



/* =====================================================
   RESULT PAGE
===================================================== */

const resultImage =
    document.getElementById(
        "resultImage"
    );


/*
 * If we're on result.html,
 * load the stored API response.
 */

if (resultImage) {

    loadResultPage();

}



function loadResultPage() {


    const storedResult =
        sessionStorage.getItem(
            "predictionResult"
        );


    /*
     * No result means the user
     * opened result.html directly.
     */

    if (!storedResult) {

        window.location.href =
            "./main.html";

        return;

    }


    let result;


    try {

        result =
            JSON.parse(
                storedResult
            );

    }

    catch {

        window.location.href =
            "./main.html";

        return;

    }



    /* -----------------------------------------
       IMAGE
    ----------------------------------------- */

    const image =
        sessionStorage.getItem(
            "uploadedImage"
        );


    if (image) {

        resultImage.src =
            image;

    }

    else {

        resultImage.style.display =
            "none";

    }



    /* -----------------------------------------
       JSON
    ----------------------------------------- */

    const jsonOutput =
        document.getElementById(
            "jsonOutput"
        );


    if (jsonOutput) {

        jsonOutput.textContent =
            JSON.stringify(
                result,
                null,
                2
            );

    }



    /* -----------------------------------------
       RETAKE
    ----------------------------------------- */

    if (
        result.status ===
        "retake"
    ) {

        renderRetakeResult(
            result
        );

        return;

    }



    /* -----------------------------------------
       NORMAL RESULT
    ----------------------------------------- */

    renderNormalResult(
        result
    );

}



/* =====================================================
   NORMAL RESULT
===================================================== */

function renderNormalResult(result) {


    const prediction =
        result.prediction ||
        {};


    const diseaseInfo =
        result.disease_info ||
        {};



    const diseaseName =
        document.getElementById(
            "diseaseName"
        );


    const scientificName =
        document.getElementById(
            "scientificName"
        );


    const plantName =
        document.getElementById(
            "plantName"
        );


    const confidence =
        document.getElementById(
            "confidence"
        );


    const confidenceText =
        document.getElementById(
            "confidenceText"
        );


    const confidenceFill =
        document.getElementById(
            "confidenceFill"
        );


    const description =
        document.getElementById(
            "description"
        );



    /* Disease */

    if (diseaseName) {

        diseaseName.textContent =
            prediction.disease ||
            diseaseInfo.name ||
            "Healthy crop";

    }



    /* Plant */

    if (plantName) {

        plantName.textContent =
            prediction.plant ||
            "Unknown";

    }



    /*
     * Current API may return class_label.
     *
     * We display it only if present.
     */

    if (scientificName) {

        scientificName.textContent =
            prediction.class_label ||
            "";

    }



    /* Confidence */

    const confidencePercent =
        Number(
            prediction.confidence_percent ||
            (
                Number(
                    prediction.confidence
                ) * 100
            ) ||
            0
        );


    if (confidence) {

        confidence.textContent =
            confidencePercent.toFixed(2)
            + "%";

    }


    if (confidenceText) {

        confidenceText.textContent =
            confidencePercent.toFixed(2)
            + "%";

    }


    if (confidenceFill) {

        confidenceFill.style.width =
            Math.min(
                confidencePercent,
                100
            ) + "%";

    }



    /* Status */

    const resultStatus =
        document.getElementById(
            "resultStatus"
        );


    if (resultStatus) {

        if (
            result.status ===
            "healthy"
        ) {

            resultStatus.textContent =
                "● Healthy crop";

        }

        else {

            resultStatus.textContent =
                "● Disease detected";

        }

    }



    /* Description */

    if (description) {

        description.textContent =
            diseaseInfo.description ||
            "No description is available for this prediction.";

    }



    /* Lists */

    renderList(
        "symptoms",
        diseaseInfo.symptoms
    );


    renderList(
        "causes",
        diseaseInfo.causes
    );


    renderList(
        "actions",
        diseaseInfo.recommended_action
    );


    renderList(
        "prevention",
        diseaseInfo.prevention
    );



    /* Top predictions */

    renderTopPredictions(
        result.top_predictions
    );

}



/* =====================================================
   RETAKE RESULT
===================================================== */

function renderRetakeResult(result) {


    const resultStatus =
        document.getElementById(
            "resultStatus"
        );


    const diseaseName =
        document.getElementById(
            "diseaseName"
        );


    const scientificName =
        document.getElementById(
            "scientificName"
        );


    const plantName =
        document.getElementById(
            "plantName"
        );


    const confidence =
        document.getElementById(
            "confidence"
        );


    const confidenceText =
        document.getElementById(
            "confidenceText"
        );


    const confidenceFill =
        document.getElementById(
            "confidenceFill"
        );


    const description =
        document.getElementById(
            "description"
        );



    if (resultStatus) {

        resultStatus.textContent =
            "● Image needs another photo";

        resultStatus.style.background =
            "#fff1ee";

        resultStatus.style.color =
            "#a63d31";

    }


    if (diseaseName) {

        diseaseName.textContent =
            "Clearer image needed";

    }


    if (scientificName) {

        scientificName.textContent =
            "";

    }


    if (plantName) {

        plantName.textContent =
            "—";

    }


    const percent =
        Number(
            result.confidence_percent ||
            0
        );


    if (confidence) {

        confidence.textContent =
            percent.toFixed(2) +
            "%";

    }


    if (confidenceText) {

        confidenceText.textContent =
            percent.toFixed(2) +
            "%";

    }


    if (confidenceFill) {

        confidenceFill.style.width =
            percent + "%";

    }


    if (description) {

        description.textContent =
            result.message ||
            "The image was not clear enough for a reliable prediction. Please upload another image.";

    }


    renderList(
        "symptoms",
        [
            "Use better lighting.",
            "Keep the affected leaf in focus.",
            "Make sure the affected area is visible."
        ]
    );


    renderList(
        "causes",
        [
            "The model confidence was below the required threshold."
        ]
    );


    renderList(
        "actions",
        [
            "Take another clear photo of the crop leaf."
        ]
    );


    renderList(
        "prevention",
        [
            "Avoid blurry or extremely dark images."
        ]
    );


    const top =
        document.getElementById(
            "topPredictions"
        );


    if (top) {

        top.innerHTML =
            "";

    }

}



/* =====================================================
   LIST RENDERER
===================================================== */

function renderList(
    elementId,
    values
) {


    const element =
        document.getElementById(
            elementId
        );


    if (!element) {
        return;
    }


    element.innerHTML =
        "";


    if (
        !Array.isArray(values) ||
        values.length === 0
    ) {

        element.innerHTML =
            "<li>No information available.</li>";

        return;

    }


    values.forEach(
        value => {

            const li =
                document.createElement(
                    "li"
                );


            li.textContent =
                value;


            element.appendChild(
                li
            );

        }
    );

}



/* =====================================================
   TOP PREDICTIONS
===================================================== */

function renderTopPredictions(
    predictions
) {


    const container =
        document.getElementById(
            "topPredictions"
        );


    if (!container) {
        return;
    }


    container.innerHTML =
        "";


    if (
        !Array.isArray(predictions) ||
        predictions.length === 0
    ) {

        container.innerHTML =
            "<p>No additional predictions available.</p>";

        return;

    }


    predictions.forEach(
        (
            prediction,
            index
        ) => {


            const row =
                document.createElement(
                    "div"
                );


            row.className =
                "prediction-row";


            const confidence =
                Number(
                    prediction.confidence_percent ||
                    (
                        Number(
                            prediction.confidence
                        ) * 100
                    ) ||
                    0
                );


            row.innerHTML = `

                <div class="prediction-rank">
                    ${index + 1}
                </div>

                <div>

                    <div class="prediction-name">

                        ${
                            prediction.disease ||
                            prediction.class_label ||
                            "Unknown"
                        }

                    </div>

                    <div class="prediction-plant">

                        ${
                            prediction.plant ||
                            "Unknown crop"
                        }

                    </div>

                </div>

                <div class="prediction-confidence">

                    ${confidence.toFixed(2)}%

                </div>

            `;


            container.appendChild(
                row
            );

        }
    );

}



/* =====================================================
   JSON TOGGLE
===================================================== */

const jsonToggle =
    document.getElementById(
        "jsonToggle"
    );


if (jsonToggle) {

    jsonToggle.addEventListener(
        "click",
        () => {


            const jsonOutput =
                document.getElementById(
                    "jsonOutput"
                );


            const visible =
                jsonOutput.style.display ===
                "block";


            jsonOutput.style.display =
                visible
                    ? "none"
                    : "block";


            jsonToggle.textContent =
                visible
                    ? "Show JSON"
                    : "Hide JSON";

        }
    );

}



/* =====================================================
   NEW ANALYSIS
===================================================== */

const newAnalysis =
    document.getElementById(
        "newAnalysis"
    );


if (newAnalysis) {

    newAnalysis.addEventListener(
        "click",
        () => {

            sessionStorage.removeItem(
                "predictionResult"
            );

            sessionStorage.removeItem(
                "uploadedImage"
            );

            window.location.href =
                "./main.html";

        }
    );

}
