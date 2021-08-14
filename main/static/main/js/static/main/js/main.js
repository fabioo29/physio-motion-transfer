$(document).ready(function(){
    $(".fancybox").fancybox({
          openEffect: "none",
          closeEffect: "none"
      });
      
      $(".zoom").hover(function(){
          
          $(this).addClass('transition');
      }, function(){
          
          $(this).removeClass('transition');
      });
      
      if (window.location.href.indexOf("loading") > -1) {
        document.location.href = location.protocol + '//' + location.host + '/getVideo/';
      }

      if (window.location.href.indexOf("getVideo") > -1) {
        document.location.href = location.protocol + '//' + location.host + '/playVideo/';
      }

  });

document.querySelector("html").classList.add('js');

window.onload=function(){
    var fileInput  = document.querySelector('.input-file.if-image'),
        fileInput2  = document.querySelector('.input-file.if-video'),
        button  = document.querySelector('.input-file-trigger.if-image'),
        button2  = document.querySelector('.input-file-trigger.if-video')
    
    button.addEventListener( "click", function( event ) {
    fileInput.focus();
    return false;
    });

    fileInput.addEventListener( "change", function( event ) {
        document.getElementById("addImage").submit();
    });

    button2.addEventListener( "keydown", function( event ) {  
        if ( event.keyCode == 13 || event.keyCode == 32 ) {  
            fileInput2.focus(); 
        }  
    });

    button2.addEventListener( "click", function( event ) {
    fileInput2.focus();
    return false;
    });  
    
    fileInput2.addEventListener( "change", function( event ) {
        document.getElementById("addVideo").submit();
    });
} 

setInterval(() => {$('#toReload').load(" #toReload");}, 5000 );
setInterval(() => {$('#pb-ui').load(" #pb-ui > *");}, 1000 );