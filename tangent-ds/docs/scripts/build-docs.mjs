import {spawn} from 'node:child_process';

const npmCommand = process.platform === 'win32' ? 'npm.cmd' : 'npm';

function run(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: 'inherit',
    });
    child.on('close', code => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`${command} ${args.join(' ')} exited with code ${code}`));
      }
    });
    child.on('error', reject);
  });
}

async function main() {
  if (process.env.SKIP_GENERATE_API !== '1') {
    await run(npmCommand, ['run', '--silent', 'generate:api']);
  }
  await run(npmCommand, ['run', '--silent', 'docusaurus', '--', 'build']);
}

await main();
