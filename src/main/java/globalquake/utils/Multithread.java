package globalquake.utils;

public class Multithread {

	public Multithread() {

	}

	int running = 0;

	public void superSplit(MultithreadSection s, int threads, int min, int max, boolean wait) {
		Object o = new Object();
		running=threads;
		for (int i = 0; i < threads; i++) {
			int start = (int) Math.floor(min + (max - min + 1) * (i / (double) (threads)));
			int end = (int) Math.floor(min + (max - min + 1) * ((i + 1) / (double) (threads))) - 1;
			int i2 = i;
			new Thread() {
				public void run() {
					s.init(i2);
					for (int j = start; j <= end; j++) {
						try {
							s.run(i2, start, end, j);
						} catch (Exception e) {
							e.printStackTrace();
						}

					}
					s.end(i2);
					if (wait) {
						running--;
						if (running == 0) {
							synchronized (o) {
								o.notify();
							}
						}
					}
				};
			}.start();
		}
		if (wait) {
			synchronized (o) {
				try {
					o.wait();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
		}
	}
}
