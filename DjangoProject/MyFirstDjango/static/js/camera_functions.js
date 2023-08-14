// camera_functions.js

// 獲取相關的DOM元素
document.addEventListener('DOMContentLoaded', () => {
    // 獲取相關的DOM元素
    const camera = document.getElementById('camera');
    const shootButton = document.getElementById('shoot');
    const chooseImageButton = document.getElementById('choose_img');
    const fileInput = document.getElementById('file');
    const base64FileInput = document.getElementById('base64_file');

    // 檢查是否有可用的攝影機
    navigator.mediaDevices.enumerateDevices()
        .then(devices => {
            const hasCamera = devices.some(device => device.kind === 'videoinput');
            if (!hasCamera) {
                shootButton.disabled = true;
                shootButton.title = 'No camera available';
            }
        })
        .catch(error => {
            console.error('Error enumerating devices:', error);
        });

    // 拍照功能
    shootButton.addEventListener('click', () => {
        if (shootButton.disabled) {
            alert('No camera available');
            return;
        }

        const canvas = document.createElement('canvas');
        canvas.width = camera.videoWidth;
        canvas.height = camera.videoHeight;
        canvas.getContext('2d').drawImage(camera, 0, 0, canvas.width, canvas.height);

        // 壓縮影像
        const compressedBase64Data = compressImage(canvas, 0.7); // 設定壓縮品質，0.7表示70%品質
        base64FileInput.value = compressedBase64Data;

        // 提交表單
        image.submit();
    });

    // 選擇照片功能
    fileInput.addEventListener('change', () => {
        const selectedFile = fileInput.files[0];
        if (selectedFile) {
            const reader = new FileReader();
            reader.onload = event => {
                const img = new Image();
                img.src = event.target.result;
                
                img.onload = () => {
                    const canvas = document.createElement('canvas');
                    const maxWidth = 800; // 最大寬度
                    const scale = img.width / maxWidth;
                    canvas.width = maxWidth;
                    canvas.height = img.height / scale;
                    canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
                    
                    // 壓縮影像
                    const compressedBase64Data = compressImage(canvas, 0.7); // 設定壓縮品質，0.7表示70%品質
                    base64FileInput.value = compressedBase64Data;
                    
                    // 提交表單
                    image.submit();
                };
            };
            reader.readAsDataURL(selectedFile);
        }
    });

    // 壓縮影像函式
    function compressImage(canvas, quality) {
        const compressedData = canvas.toDataURL('image/jpeg', quality);
        return compressedData;
    }

});
