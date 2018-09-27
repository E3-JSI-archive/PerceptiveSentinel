// import qm module
var qm = require('qminer');
// create a simple base containing one store
var base = new qm.Base({
   mode: "createClean",
   schema: [{
       name: "People",
       fields: [
           { name: "Name", type: "string" },
           { name: "Gendre", type: "string" },
       ]
   },
   {
       name: "Laser",
       fields: [
           { name: "Time", type: "datetime" },
           { name: "WaveLength", type: "float" }
       ]
   }]
});

// create a new stream aggregator for 'People' store, get the length of the record name (with the function object)
var aggr = new qm.StreamAggr(base, new function () {
   var numOfAdds = 0;
   var numOfUpdates = 0;
   var numOfDeletes = 0;
   var time = "";
   this.name = 'nameLength',
   this.onAdd = function (rec) {
       numOfAdds += 1;
   };
   this.onUpdate = function (rec) {
       numOfUpdates += 1;
   };
   this.onDelete = function (rec) {
       numOfDeletes += 1;
   };
   this.onTime = function (ts) {
       time = ts;
   };
   this.saveJson = function (limit) {
       return { adds: numOfAdds, updates: numOfUpdates, deletes: numOfDeletes, time: time };
   };
}, "People");

console.log()