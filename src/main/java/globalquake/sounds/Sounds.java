package globalquake.sounds;

import globalquake.core.exception.FatalIOException;
import globalquake.core.Settings;
import org.tinylog.Logger;

import javax.sound.sampled.*;
import java.util.Objects;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Sounds {

	public static Clip weak;
	public static Clip moderate;
	public static Clip shindo5;
	public static Clip warning;
	public static Clip intensify;
	public static Clip eew_warning;
	public static Clip felt;
	public static Clip dong;
	public static Clip update;

	public static Clip found;

	public static boolean soundsAvailable = true;

	private static final ExecutorService soundService = Executors.newSingleThreadExecutor();

	private static Clip loadSound(String res) throws FatalIOException {
		try {
			AudioInputStream audioIn = AudioSystem.getAudioInputStream(Objects.requireNonNull(ClassLoader.getSystemClassLoader().getResource(res)));
			Clip clip = AudioSystem.getClip();
			clip.open(audioIn);
			return clip;
		} catch(Exception e){
			soundsAvailable = false;
			throw new FatalIOException("Failed to load sound: "+res, e);
		}
	}

	public static void load() throws Exception {
		weak = loadSound("sounds/weak.wav");
		moderate = loadSound("sounds/moderate.wav");
		shindo5 = loadSound("sounds/shindo5.wav");
		intensify = loadSound("sounds/intensify.wav");
		found = loadSound("sounds/found.wav");
		update = loadSound("sounds/update.wav");
		warning = loadSound("sounds/warning.wav");
		eew_warning = loadSound("sounds/eew_warning.wav");
		felt = loadSound("sounds/felt.wav");
		dong = loadSound("sounds/dong.wav");

	}

	public static void playSound(Clip clip) {
		if(!Settings.enableSound || !soundsAvailable || clip == null) {
			return;
		}

		soundService.submit(new Runnable() {
			@Override
			public void run() {

				var latch = new CountDownLatch(1);
				clip.addLineListener(new LineListener() {
					@Override
					public void update(LineEvent event) {
						if (event.getType().equals(LineEvent.Type.STOP)) {
							clip.removeLineListener(this);
							latch.countDown();
						}
					}
				});
				clip.setFramePosition(0);
				clip.start();
				try {
					latch.await();
				} catch (InterruptedException e) {
					Logger.error(e);
				}
			}
		});
	}
	
}
