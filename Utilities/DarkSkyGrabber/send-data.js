var kafka = require('kafka-node');
var fs = require('fs')
var HighLevelProducer = kafka.HighLevelProducer;
// var client = new kafka.Client();
const client = new kafka.Client("localhost:2181");
var producer = new HighLevelProducer(client);


var text = fs.readFileSync('data_Aarhus.json', 'utf8');
text = text.replace(/\n/g, ',')
text = text.substring(0,text.length-1)

var json_data = JSON.parse('['+text+']');
//send = JSON.stringify(json_data[0]);
//console.log(JSON.stringify(json_data[0]));



for(dat of json_data){
    dat = JSON.stringify(dat)
    var payloads = [
        { topic: 'test', messages: dat },
    // { topic: 'measurements_node_N1', messages: ['hello', 'world'] }
    ];


    producer.on('ready', function () {
        producer.send(payloads, function (err, data) {
            console.log("DAta", data);
        });
    });
}

/* let Consumer = kafka.Consumer;        
//this.client = new kafka.Client(this.connectionConfig.zookeeper, this.id);
// this.offset = new kafka.Offset(this.client);
let consumer = new Consumer(
    client,
    [ { topic: "test", partition: 0 }],
    { groupId: "test" }
);

consumer.on('message', function (message) {
    try {
        console.log(message);
    } catch (err) {
        console.log("ERROR", err);
    }
});

consumer.on('error', function (err) {
    console.log("Error", err);
});
 */