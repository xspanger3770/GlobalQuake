package globalquake.sounds;

import globalquake.core.exception.FatalIOException;

import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.Clip;
import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.Map;

import static globalquake.sounds.Sounds.EXPORT_DIR;

public class GQSound {

    private final String filename;
    private final String description;

    private Clip clip;

    public double volume;

    public static final Map<String, String> descriptions = new HashMap<>();

    static {
        descriptions.put("level_0.wav",
                "Triggered once so-called Cluster is created. " +
                        "\nThis happens if 4 or more stations detect shaking in close proximity.");

        descriptions.put("level_1.wav",
                "Triggered when cluster reaches level 1. " +
                        "\nThis happens when at least 7 stations reach 64 counts or at least 4 stations reach 1,000 counts.");

        descriptions.put("level_2.wav",
                "Triggered when cluster reaches level 2. " +
                        "\nThis happens when at least 7 stations reach 1,000 counts or at least 3 stations reach 10,000 counts.");

        descriptions.put("level_3.wav",
                "Triggered when cluster reaches level 3. " +
                        "\nThis happens when at least 5 stations reach 10,000 counts or at least 3 stations reach 50,000 counts.");

        descriptions.put("level_4.wav",
                """
                        Triggered when cluster reaches level 4.\s
                        This happens when at least 4 stations reach 50,000 counts.\s
                        This audio file is BLANK at default since this alarm sound has not yet been added!""");

        descriptions.put("intensify.wav", "Triggered if the conditions specified in the previous Alerts settings tab are met.");
        descriptions.put("felt.wav", "Triggered if shaking is expected at your home location." +
                "\nThe threshold intensity scale and level can be configured in the Alerts tab.");
        descriptions.put("felt_strong.wav", """
                Triggered if STRONG shaking is expected at your home location.
                The threshold intensity scale and level can be configured in the Alerts tab.
                This audio file is BLANK at default since this alarm sound has not yet been added!""");

        descriptions.put("eew_warning.wav", """
                Triggered if there is high certainty in the detected earthquake and\s
                it has at least MMI VI estimated intensity on land.
                This audio file is BLANK at default since this alarm sound has not yet been added!""");

        descriptions.put("countdown.wav", "Countdown of the last 10 seconds before S waves arrives at your home location\n" +
                "if shaking is expected there.");

        descriptions.put("update.wav", "Earthquake parameters updated (revision).");
        descriptions.put("found.wav", "Earthquake epicenter determined for the first time and it appears on the map.");
    }

    public GQSound(String filename) {
        this(filename, descriptions.getOrDefault(filename, "[No description provided]"));
    }

    public GQSound(String filename, String description) {
        this.filename = filename;
        this.description = description;
        this.volume = 1.0;
    }

    public void load(boolean externalOnly) throws FatalIOException {
        try {
            // try to load from export folder
            Path soundPath = Paths.get(EXPORT_DIR.getAbsolutePath(), filename);
            InputStream audioInStream = Files.exists(soundPath) || externalOnly ?
                    new FileInputStream(soundPath.toFile()) :
                    ClassLoader.getSystemClassLoader().getResourceAsStream("sounds/" + filename);

            if (audioInStream == null) {
                throw new IOException("Sound file not found: %s (from file = %s)".formatted(filename, Files.exists(soundPath)));
            }

            AudioInputStream audioIn = AudioSystem.getAudioInputStream(
                    new BufferedInputStream(audioInStream));
            Clip clip = AudioSystem.getClip();
            clip.open(audioIn);
            this.clip = clip;
        } catch (Exception e) {
            throw new FatalIOException("Failed to load sound: " + filename, e);
        }
    }

    public void export(Path exportPath) throws IOException {
        Path exportedFilePath = exportPath.resolve(filename);
        if (!Files.exists(exportedFilePath)) { // Check if the file already exists
            InputStream is = ClassLoader.getSystemClassLoader().getResourceAsStream("sounds/" + filename);
            if (is != null) {
                Files.copy(is, exportedFilePath, StandardCopyOption.REPLACE_EXISTING);
                is.close();
            }
        }
    }

    @SuppressWarnings("unused")
    public String getDescription() {
        return description;
    }

    public String getFilename() {
        return filename;
    }

    public Clip getClip() {
        return clip;
    }
}
