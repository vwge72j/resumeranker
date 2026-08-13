/**
 * Client-side Script for AI Resume Screening & Ranking System.
 *
 * This script provides:
 * 1. Client-side file extension validation (.txt only) before form submission.
 * 2. Visual feedback by changing the button text to "Analyzing..." during processing.
 * 3. Automatic smooth scrolling to the Result Card after analysis completes.
 */

document.addEventListener("DOMContentLoaded", function () {
    const uploadForm = document.getElementById("uploadForm");
    const resumeInput = document.getElementById("resume");
    const jdInput = document.getElementById("jd");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const resultSection = document.getElementById("resultSection");

    // Helper function to validate if a selected file ends with .txt
    function isTxtFile(fileInput) {
        if (!fileInput.value) return false;
        const fileName = fileInput.value.toLowerCase();
        return fileName.endsWith(".txt");
    }

    // Event Listener for form submission
    if (uploadForm) {
        uploadForm.addEventListener("submit", function (event) {
            // Step 1: Validate Resume Input
            if (!isTxtFile(resumeInput)) {
                event.preventDefault(); // Stop form submission
                alert("Please select a valid plain text (.txt) file for the Resume.");
                resumeInput.focus();
                return;
            }

            // Step 2: Validate Job Description Input
            if (!isTxtFile(jdInput)) {
                event.preventDefault(); // Stop form submission
                alert("Please select a valid plain text (.txt) file for the Job Description.");
                jdInput.focus();
                return;
            }

            // Step 3: Provide UI loading feedback while Flask & ML pipeline process files
            if (analyzeBtn) {
                analyzeBtn.disabled = true;
                analyzeBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-2"></i>Analyzing via ML...';
                // Allow the form to submit naturally to the backend
                uploadForm.submit();
            }
        });
    }

    // Step 4: If a Result Card is present on page load, smoothly scroll down to it
    if (resultSection) {
        resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }
});