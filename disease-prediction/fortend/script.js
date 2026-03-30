async function predict() {
    const data = {
        age: Number(age.value),
        sex: Number(sex.value),
        cp: Number(cp.value),
        trestbps: Number(trestbps.value),
        chol: Number(chol.value),
        fbs: Number(fbs.value),
        restecg: Number(restecg.value),
        thalach: Number(thalach.value),
        exang: Number(exang.value),
        oldpeak: Number(oldpeak.value)
    };

    const res = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(data)
    });

    const result = await res.json();

    document.getElementById("result").innerHTML = `
        <h3>${result.message}</h3>
        <p>Confidence: ${result.probability}%</p>
    `;
}