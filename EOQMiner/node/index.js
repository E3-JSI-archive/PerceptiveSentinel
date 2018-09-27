// EOQMiner, (c) Jozef Stefan Institute, 2018 - 2019

// STDIN API multiplexer
// TODO:
process.stdin.on('readable', () => {
    const chunk = process.stdin.read();
    if (chunk !== null) {
        process.stdout.write(`data: ${chunk}`);
    }
});

// finishing the STDIN/STDOUT API interface
process.stdin.on('end', () => {
    process.stdout.write('end');
});