async function upload() {
    let file = document.getElementById("video").files[0];

    let formData = new FormData();
    formData.append("file", file);

    let res = await fetch("http://127.0.0.1:8001/analyze", {
        method: "POST",
        body: formData
    });

    let out = await res.json();
    document.getElementById("output").innerText = JSON.stringify(out, null, 2);
}
