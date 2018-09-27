// simple echo script

process.stdout.write("Test output from simple echo.");

process.stdin.on('readable', () => {
    const chunk = process.stdin.read();    
    if (chunk !== null) {
        process.stdout.write(`data: ${chunk}`);
    }
});

process.stdin.on('end', () => {
    process.stdout.write('end');
});