var kafka = require('kafka-node');
var fs = require('fs')
var HighLevelProducer = kafka.HighLevelProducer;
// var client = new kafka.Client();
const client = new kafka.Client("localhost:2181");
var producer = new HighLevelProducer(client);


var text = fs.readFileSync('DarkSkyGrabber\\data\\data_Den_Helder_array.json', 'utf8');
var json_data = JSON.parse(text);

var time = 0;
var delay = 1000;

for(dat of json_data){
    // Directly evaluate closure
    (function (dat2){
        time = time + delay;
        setTimeout(function(){
            //console.log(dat2)
            var topics = [
                { topic: 'test', messages: dat2 }
            ];

        
               // producer.on('ready', function () {
                    producer.send(topics, function (err, data) {
                        console.log("DAta", data);
                    });
               // });
            },time);
    })(JSON.stringify(dat))
}
