document.getElementById("form").addEventListener("submit", async function(e) {
    e.preventDefault();

    // 🎯 Get values
    const Hours_Studied = Number(document.getElementById("Hours_Studied").value);
    const Attendance = Number(document.getElementById("Attendance").value);
    const Sleep_Hours = Number(document.getElementById("Sleep_Hours").value);
    const Previous_Scores = Number(document.getElementById("Previous_Scores").value);
    const Motivation_Level = document.getElementById("Motivation_Level").value;

    // ❌ Validation
    if (
        !Hours_Studied || 
        !Attendance || 
        !Sleep_Hours || 
        !Previous_Scores
    ) {
        alert("⚠️ Please fill all fields correctly!");
        return;
    }

    const data = {
        Hours_Studied,
        Attendance,
        Sleep_Hours,
        Previous_Scores,
        Motivation_Level
    };

    // ⏳ Show loading
    document.getElementById("result").innerText = "⏳ Predicting...";

    try {
        const res = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(data)
        });

        const result = await res.json();

        // ✅ Success
        if (result.predicted_score !== undefined) {
            document.getElementById("result").innerText =
                "✅ Predicted Score: " + result.predicted_score;
        } else {
            document.getElementById("result").innerText =
                "❌ Error: " + (result.message || "Something went wrong");
        }

    } catch (error) {
        // ❌ Server error
        document.getElementById("result").innerText =
            "❌ Server not running or connection error!";
    }
});