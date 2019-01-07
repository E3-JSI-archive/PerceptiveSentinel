var kafka = require('kafka-node');
const client = new kafka.Client("localhost:2181");

let Consumer = kafka.Consumer;        

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
 