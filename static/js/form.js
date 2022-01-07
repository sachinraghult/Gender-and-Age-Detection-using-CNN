$('.form button').on('click',function(){
  
    $('.belt').addClass('transitioned-1');
    $('.progress-inner').addClass('loading');
    
    setTimeout(function(){
      $('.form-container').addClass('transitioned-3');
    },3000);
    
    setTimeout(function(){
      
      $('.belt').removeClass('transitioned-1');
      $('.belt').addClass('transitioned-2');
      
    },3300);
    
    $('.success button').on('click',function(){
      
      $('.progress-inner').removeClass('loading');
      $('.belt').removeClass('transitioned-2');
      $('.form-container').removeClass('transitioned-3');
      
    });
    
    // setTimeout(function(){
    //   $('.form-container').addClass('transitioned-3');
    // },6000);
    // $('.belt').removeClass('transitioned-2');
    // $('.form-container').removeClass('transitioned-3');
  });