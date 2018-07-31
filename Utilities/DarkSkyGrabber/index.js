// DISCLAIMER: darksky.net returns hourly data just for the current day (from midnight to midnight)
// therefore there is no use for calling the script for every hour.

// includes
const request = require("sync-request");
const jsonfile = require('jsonfile');
const fs = require('fs');

// loading configuration files
const config = jsonfile.readFileSync('config.json');

const coordinate = jsonfile.readFileSync('coordinate.json');
place = 'Aarhus' // 'Grad_Jablje' or 'Aarhus' or 'Den_Helder'

// setting up environmental variables regarding time
process.env.TZ = 'Europe/Ljubljana'
let a = new Date("Fri Jul 20 2018 00:00:00 GMT+0100(CET)");
//let a = new Date("Fri Jan 1 2016 00:00:00 GMT+0100(CET)");
let timeStamp = Math.floor(a.getTime()/1000);      
console.log(timeStamp);
console.log(process.env.TZ)
let timeStampNOW = Math.floor((Date.now()+7200000)/1000);

// if it exists, read the last written timestamp
if (fs.existsSync("data_" + place + ".json")) {
    console.log("Reading last timestamp!")
    let lines = fs.readFileSync("data_" + place + ".json").toString().split("\n");
    console.log(lines[lines.length - 2]);
    let line = JSON.parse(lines[lines.length - 2]);
    timeStamp = line.daily.data[0].time;
    // add an hour
    timeStamp = timeStamp + 24*3600;
}

// gathering data until today
console.log(timeStampNOW)
//console.log(timeStamp)

// starting the loop
let latitude = coordinate[place].latitude;
let longitude = coordinate[place].longitude;
let token = config["darkSky-token"];
var array_of_data = [];
 
while (timeStamp <= timeStampNOW) {
    var date = new Date(timeStamp * 1000);
    console.log(date);    

    var url = "https://api.darksky.net/forecast/" + token + "/" + 
    latitude +"," + 
    longitude + "," + 
    timeStamp + 
    '?exclude=currently,minutely,hourly,alerts,flags'+ //only daily
    '&units=si';  //units

    var res = request("GET", url);

    fs.appendFileSync("data_" + place + ".json", res.getBody('utf8') + "\n");
   // fs.appendFileSync("data_" + place + "2.json", res.getBody('utf8') + "\n");
    console.log(JSON.parse(res.getBody('utf8')).daily.data[0]);  

    array_of_data.push(JSON.parse(res.getBody('utf8')).daily.data[0])
      
    // adding a day to the data
    timeStamp = timeStamp + 24 * 3600;
}




 { time: 1532728800,
    summary: 'Rain starting in the evening.',
    icon: 'rain',
    sunriseTime: 1532747776,
    sunsetTime: 1532806650,
    moonPhase: 0.52,
    precipIntensity: 0.315,
    precipIntensityMax: 3.2588,
    precipIntensityMaxTime: 1532811600,
    precipProbability: 0.56,
    precipType: 'rain',
    temperatureHigh: 29.76,
    temperatureHighTime: 1532786400,
    temperatureLow: 17.44,
    temperatureLowTime: 1532836800,
    apparentTemperatureHigh: 30.28,
    apparentTemperatureHighTime: 1532782800,
    apparentTemperatureLow: 17.67,
    apparentTemperatureLowTime: 1532836800,
    dewPoint: 17.53,
    humidity: 0.67,
    pressure: 1011.15,
    windSpeed: 3.98,
    windGust: 11.64,
    windGustTime: 1532804400,
    windBearing: 114,
    cloudCover: 0.28,
    uvIndex: 5,
    uvIndexTime: 1532775600,
    visibility: 14.63,
    ozone: 329.04,
    temperatureMin: 19.27,
    temperatureMinTime: 1532811600,
    temperatureMax: 29.76,
    temperatureMaxTime: 1532786400,
    apparentTemperatureMin: 19.64,
    apparentTemperatureMinTime: 1532811600,
    apparentTemperatureMax: 30.28,
    apparentTemperatureMaxTime: 1532782800 }