document
  .getElementById("profileForm")
  .addEventListener("submit", function (event) {
    // Show loading screen as soon as the button is clicked
    document.getElementById("contentWrapper").classList.add("hidden");
    document.getElementById("loadingScreen").classList.remove("hidden-load");
    document.getElementById("nav").classList.add("hidden");
    document.getElementById("footer").classList.add("hidden");
    // Set initial loading percentage
    let loadingPercentage = 0;
    const loadingText = document.getElementById("loadingPercentage");

    // Update the loading percentage from 0 to 100 over a set time interval
    const loadingInterval = setInterval(function () {
      if (loadingPercentage < 100) {
        loadingPercentage++;
        loadingText.textContent = loadingPercentage + "%";
      } else {
        clearInterval(loadingInterval);
      }
    }, 50); // Update every 50 milliseconds
  });

function closeModal() {
  document.getElementById("profileModal").style.display = "none";

  // Show form and hide loading screen after modal is closed
  document.getElementById("contentWrapper").classList.remove("hidden");
  document.getElementById("nav").classList.remove("hidden");
  document.getElementById("footer").classList.remove("hidden");
  document.getElementById("loadingScreen").classList.add("hidden-load");
}

document.addEventListener("DOMContentLoaded", function () {
  var modal = document.getElementById("profileModal");
  if (modal) {
    modal.style.display = "block";
    // Hide loading screen as soon as modal is shown
    document.getElementById("loadingScreen").classList.add("hidden-load");
  }
});
function toggleNavbar() {
  const navbarLinks = document.querySelector(".navbar-links");
  navbarLinks.classList.toggle("active");
}
