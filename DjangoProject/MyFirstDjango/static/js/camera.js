var constraints = {
    video: { width: 640, height: 640 }
};

var video = document.querySelector('video');

function handleSuccess(stream) {
    window.stream = stream; // only to make stream available to console
    video.srcObject = stream;
}

function handleError(error) {
    console.log('getUserMedia error: ', error);
}

navigator.mediaDevices.getUserMedia(constraints).
    then(handleSuccess).catch(handleError);

$('#shoot').click(function(){
    shoot();
    $('.hidden_block').css("display","block");
    var img = $('canvas')[0].toDataURL("image/jpeg");
    $('#base64_file').val(img);
});
$('#back').click(function(){
    $('.hidden_block').css("display","none");
});
$('#go').click(function(){
    //$('canvas').toDataUrl(image/jpeg);
    form_submit();
});
$('#file').change(function(){
    form_submit();
});
function form_submit(){
    $('#image').submit();
}
function shoot() {
    var video = $("#camera")[0];
    var canvas = capture(video);
    $("#result").empty();
    $("#result").append(canvas);
    $('canvas').css("width","100%");
    $('canvas').css("aspect-ratio"," 1 / 1");
}
function capture(video) {
    var canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    var ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    return canvas;
}
