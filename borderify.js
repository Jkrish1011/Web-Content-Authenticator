
var mm = $("#rso");
mm.css("border","5px solid red");


var ss = mm.find("a");

var ls = {};
var ls1 = {};
var k = 0;
var hf;
var chck = new RegExp("https://webcache");
var chck1 = new RegExp("http://webcache");

function isUndefined(value){
   
    var undefined = void(0);
    return value === undefined;


}

function insert_Func(hf){
	var i,flag=0,len;

	if(Object.getOwnPropertyNames(ls).length === 0){
		return true;
	}
	else{
		len = Object.getOwnPropertyNames(ls).length;
		for(i=0;i<len;i++){
			if(hf === ls[i]){
				return false;
			}
			else{
				if(hf.includes(ls[i]))
					return false;
			}
		}


	}
	
		return true;

}

$('#rso').find('a').each(function() {
	if(isUndefined($(this).attr('href') )){}
		
		else{
		hf = "" + $(this).attr('href');
		if(hf[0] != '#' && hf[0] != '/'){
			
			if(chck.test(hf)){}
				else if(chck1.test(hf)){}
					else{
						var kk = insert_Func(hf);
						if(kk){
							ls[k] = hf;
							k++;
						}
					}
		}
	}
});

function display_Score(msg){
// for(var i =0;i<len;i++){
// 	var he = 'a[href^=\"' + ls[i] +'\"]';
// 	console.log(he);

// 	$(he).css("backgroundcolor","blue");
// 		}
var k=0;

$('#rso').find('a').each(function() {
	if(isUndefined($(this).attr('href') )){}
		
		else{
		hf = "" + $(this).attr('href');
		if(hf[0] != '#' && hf[0] != '/'){
			
			if(chck.test(hf)){}
				else if(chck1.test(hf)){}
					else{
						// $(this).css("color","red");
						// var $e = $("<div>", {id: "newDiv1", style:"backgroundcolor=red;width=10px;height=10px;"});
						$(this).append(
						  $('<div/>')
						    .attr("style", "position:relative;width:200px;height:20px;background:green;font:10px;")
						    .text(msg[k])
						);
						k++;
						// if(hf == (ls[k])){
						// 	// $(this).css("backgroundcolor","blue");
						// console.log(ls[k]);	
							
						// }
						// k++;

					}
		}
	}
});



}
$(document).ready(function(){
	var request = $.ajax({
  url: "http://127.0.0.1:9000/plugin/api/plugin",
  method: "POST",
  contentType: "application/json;charset=utf-8;",
   dataType: "json",
  data: JSON.stringify(ls),
});
 
request.done(function( msg ) {
	
   display_Score(msg);

});
 
request.fail(function( jqXHR, textStatus ) {
  alert( "Request failed: " + textStatus );
  console.log(jqXHR);
});

});




   
