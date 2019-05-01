
var xhr = new XMLHttpRequest();
xhr.open("POST", "http://127.0.0.1:5000/start", true)
xhr.send()
xhr.onload = function(e){
  document.getElementById("load").innerHTML = "System Ready"
  console.log(xhr.responseText)
  document.getElementById("submit_tweet").disabled = false
}



function submit_tweet(){
  var id = document.getElementById("tweet_id").value
  data = {"tid":id}
  console.log(data)
  var xhr = new XMLHttpRequest();
  xhr.open("POST", "http://127.0.0.1:5000/set_tid", true)
  xhr.send(JSON.stringify(data))
  xhr.onload = function(e){
    console.log(xhr.responseText)
    get_tweet()
  }
}

function submit_user(){
  var id = document.getElementById("user_id").value
  data = {"uid":id}
  console.log(data)
  var xhr = new XMLHttpRequest();
  xhr.open("POST", "http://127.0.0.1:5000/set_uid", true)
  xhr.send(JSON.stringify(data))
  xhr.onload = function(e){
    console.log(xhr.responseText)
  }
}

function get_tweet(){
  var xhr = new XMLHttpRequest();
  xhr.open("GET", "http://127.0.0.1:5000/get_tweet", true)
  xhr.send()
  xhr.onload = function(e){
    console.log(xhr.responseText)
    var json = JSON.parse(xhr.responseText)
    console.log(json)
    display_tweet(json.name, json.text)
    if(json.is_bot == 1){
      get_user()
      tweet_text("This tweet is highly likely to come from a bot")
    } else {
      tweet_text("This tweet does not come from a bot")
      profile_text("This profile is not a bot")
      get_user()
    }
  }
}

function display_tweet(name, text){
  document.getElementById("name").innerHTML = name
  document.getElementById("tweet").innerHTML = text
}

function get_user(){
  var xhr = new XMLHttpRequest();
  xhr.open("GET", "http://127.0.0.1:5000/get_user", true)
  xhr.send()
  xhr.onload = function(e){
    console.log(xhr.responseText)
    var json = JSON.parse(xhr.responseText)
    if(json.is_bot == 1){
      profile_text("This profile is a bot")
    } else {
      profile_text("This profile is not a bot")
    }
  }
}

function tweet_text(new_text){
  document.getElementById("is_bot_tweet").innerHTML = new_text
}

function profile_text(new_text){
  document.getElementById("is_bot_profile").innerHTML = new_text
}
