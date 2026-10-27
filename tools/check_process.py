"""Stream bounded-check output to immutable logs; terminate our child on timeout."""
import codecs,os,signal,subprocess,sys,threading,time
from pathlib import Path


def run_checks_process(command,*,cwd,env,output,timeout=3600,heartbeat=30):
    if not 0<float(timeout)<=7200 or heartbeat<=0:raise ValueError('invalid bounded-check wall timeout')
    output=Path(output);errors=[];started=time.monotonic()
    with (output/'checks.stdout.log').open('xb') as stdout_log,(output/'checks.stderr.log').open('xb') as stderr_log:
        child=subprocess.Popen(command,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            start_new_session=os.name=='posix')
        def copy(pipe,log,terminal):
            decoder=codecs.getincrementaldecoder('utf-8')(errors='replace')
            try:
                while True:
                    block=pipe.read1(4096)
                    if not block:break
                    log.write(block);log.flush();terminal.write(decoder.decode(block));terminal.flush()
                terminal.write(decoder.decode(b'',final=True));terminal.flush()
            except Exception as error:errors.append(str(error))
            finally:pipe.close()
        threads=[threading.Thread(target=copy,args=(child.stdout,stdout_log,sys.stdout),daemon=True),
                 threading.Thread(target=copy,args=(child.stderr,stderr_log,sys.stderr),daemon=True)]
        for thread in threads:thread.start()
        timed_out=False
        def stop():
            if os.name=='posix':
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            else:child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name=='posix':
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                else:child.kill()
                child.wait(timeout=5)
        try:
            while True:
                remaining=float(timeout)-(time.monotonic()-started)
                if remaining<=0:timed_out=True;stop();break
                try:child.wait(timeout=min(heartbeat,remaining));break
                except subprocess.TimeoutExpired:
                    print(f'[validation] checks running: {time.monotonic()-started:.0f}s; logs saved live',flush=True)
        except BaseException:
            stop();raise
        finally:
            for thread in threads:thread.join(timeout=10)
        if any(thread.is_alive() for thread in threads):raise RuntimeError('check log reader did not exit')
        if errors:raise RuntimeError('check log stream failed: '+repr(errors))
    return dict(command=command,exit_code=child.returncode,timed_out=timed_out,
        elapsed_seconds=time.monotonic()-started,timeout_seconds=float(timeout),logs_written_live=True)
