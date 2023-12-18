package globalquake.sounds;

import globalquake.core.GlobalQuake;
import globalquake.core.exception.FatalIOException;
import globalquake.core.Settings;
import org.tinylog.Logger;

import javax.sound.sampled.*;
import java.io.BufferedInputStream;
import java.io.File;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Sounds {

	public static Clip level_0;
	public static Clip level_1;
	public static Clip level_2;
	public static Clip level_3;
	public static Clip intensify;
	public static Clip felt;
	public static Clip countdown;
	public static Clip update;

	public static Clip found;

	public static boolean soundsAvailable = true;

	private static final ExecutorService soundService = Executors.newCachedThreadPool();

	private static Clip loadSound(String res) throws FatalIOException {
		try {
			AudioInputStream audioIn = AudioSystem.getAudioInputStream(
					new BufferedInputStream(Objects.requireNonNull(ClassLoader.getSystemClassLoader().getResource(res)).openStream(), 65565));
			Clip clip = AudioSystem.getClip();
			clip.open(audioIn);
			return clip;
		} catch(Exception e){
			soundsAvailable = false;
			throw new FatalIOException("Failed to load sound: "+res, e);
		}
	}

	public static void load() throws Exception {
		// CLUSTER
		level_0 = loadSound("sounds/level_0.wav");
		level_1 = loadSound("sounds/level_1.wav");
		level_2 = loadSound("sounds/level_2.wav");
		level_3 = loadSound("sounds/level_3.wav");
		// lvl 4 ?

		// QUAKE
		found = loadSound("sounds/found.wav");
		update = loadSound("sounds/update.wav");

		// LOCAL
		intensify = loadSound("sounds/intensify.wav");
		felt = loadSound("sounds/felt.wav");
		// strong_felt ?

		// ???
		countdown = loadSound("sounds/countdown.wav");
	}

	public static void playSound(Clip clip) {
		if(!Settings.enableSound || !soundsAvailable || clip == null) {
			return;
		}

		soundService.submit(() -> {
			try {
				playClipRuntime(clip);
			} catch(Exception e){
				Logger.error(e);
			}
		});
	}

	private static void playClipRuntime(Clip clip) {
		clip.stop();
		clip.flush();
		clip.setFramePosition(0);
		clip.start();
        try {
            Thread.sleep(clip.getMicrosecondLength() / 1000);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

	public static void main(String[] args) throws Exception {
		GlobalQuake.prepare(new File("./"), null);
		load();
		playSound(update);
		Thread.sleep(2000);
		System.exit(0);
	}

}
