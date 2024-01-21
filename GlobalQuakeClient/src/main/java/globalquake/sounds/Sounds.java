package globalquake.sounds;

import globalquake.core.GlobalQuake;
import globalquake.core.Settings;
import globalquake.core.exception.FatalIOException;
import globalquake.core.exception.RuntimeApplicationException;
import org.tinylog.Logger;

import javax.sound.sampled.*;
import java.io.*;
import java.nio.file.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Sounds {

	public static final File EXPORT_DIR = new File(GlobalQuake.mainFolder, "sounds/");
	public static GQSound level_0 = new GQSound("level_0.wav");
	public static GQSound level_1 = new GQSound("level_1.wav");
	public static GQSound level_2 = new GQSound("level_2.wav");
	public static GQSound level_3 = new GQSound("level_3.wav");
	public static GQSound level_4 = new GQSound("level_4.wav");
	public static GQSound intensify = new GQSound("intensify.wav");
	public static GQSound felt = new GQSound("felt.wav");
	public static GQSound eew_warning = new GQSound("eew_warning.wav");
	public static GQSound felt_strong = new GQSound("felt_strong.wav");
	public static GQSound countdown = new GQSound("countdown.wav");
	public static GQSound countdown2 = new GQSound("countdown.wav");
	public static GQSound update = new GQSound("update.wav");
	public static GQSound found = new GQSound("found.wav");

	public static GQSound[] ALL_SOUNDS = {
			level_0,
			level_1,
			level_2,
			level_3,
			level_4,
			intensify,
			felt,
			eew_warning,
			felt_strong,
			countdown,
			countdown2, // workaround
			update,
			found
	};

	public static GQSound[] ALL_ACTUAL_SOUNDS = {
			level_0,
			level_1,
			level_2,
			level_3,
			level_4,
			intensify,
			felt,
			eew_warning,
			felt_strong,
			countdown,
			update,
			found
	};

	public static boolean soundsAvailable = true;

	private static final ExecutorService soundService = Executors.newCachedThreadPool();

	public static void exportSounds() throws IOException {
		Path exportPath = Paths.get(EXPORT_DIR.getAbsolutePath());
		if (!Files.exists(exportPath)) {
			Files.createDirectory(exportPath);
			writeReadmeFile(exportPath);
		}

		try {
			for (GQSound gqSound : ALL_ACTUAL_SOUNDS) {
				gqSound.export(exportPath);
			}
		} catch(IOException e){
			GlobalQuake.getErrorHandler().handleWarning(new RuntimeApplicationException("Unable to export sounds to %s!".formatted(exportPath.toString())));
		}
	}


	private static void loadSounds() {
		try {
			for (GQSound gqSound : ALL_SOUNDS) {
				gqSound.load();
			}
		} catch(FatalIOException e){
			soundsAvailable = false;
			GlobalQuake.errorHandler.handleWarning(e);
		}
	}

	private static void writeReadmeFile(Path exportPath) throws IOException {
		String readmeContent = """
                README

                This directory contains the exported sound files from GlobalQuake.
                You can edit these sound files as per your preference.
                Please note that the sounds will only be exported once, meaning that any changes you make here will be kept and used by GlobalQuake.
                After uploading your sounds, please restart GlobalQuake.

                Enjoy customizing your sound experience!""";

		Files.writeString(exportPath.resolve("README.txt"), readmeContent, StandardOpenOption.CREATE);
	}

	public static void load() throws Exception {
		exportSounds();
		loadSounds();
	}

	public static void playSound(GQSound sound) {
		if(!Settings.enableSound || !soundsAvailable || sound == null || sound.getClip() == null) {
			return;
		}

		soundService.submit(() -> {
			try {
				playClipRuntime(sound);
			} catch(Exception e){
				Logger.error(e);
			}
		});
	}

	private static void playClipRuntime(GQSound sound) {
		Clip clip = sound.getClip();
		clip.stop();
		clip.flush();
		clip.setFramePosition(0);

		double volume = Math.max(0.0, Math.min(1.0, sound.volume * (Settings.globalVolume / 100.0)));
		FloatControl gainControl = (FloatControl) clip.getControl(FloatControl.Type.MASTER_GAIN);
		gainControl.setValue(20f * (float) Math.log10(volume));

		clip.start();

        try {
            Thread.sleep(clip.getMicrosecondLength() / 1000);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
	}

	public static void main(String[] args) throws Exception {
		GlobalQuake.prepare(new File("."), null);
		load();

		playSound(level_2);

		Thread.sleep(3000);

		System.exit(0);
	}

}
