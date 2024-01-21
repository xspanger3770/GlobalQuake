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

import static globalquake.sounds.Sounds.EXPORT_DIR;

public class GQSound {

    private final String filename;
    private final String description;

    private Clip clip;

    public double volume;

    public GQSound(String filename){
        this(filename, "");
    }

    public GQSound(String filename, String description) {
        this.filename = filename;
        this.description = description;
        this.volume = 1.0; // TODO load/save
    }

    public void load() throws FatalIOException {
        try {
            // try to load from export folder
            Path soundPath = Paths.get(EXPORT_DIR.getAbsolutePath(), filename);
            InputStream audioInStream = Files.exists(soundPath) ?
                    new FileInputStream(soundPath.toFile()) :
                    ClassLoader.getSystemClassLoader().getResourceAsStream("sounds/" + filename);

            if (audioInStream == null) {
                throw new IOException("Sound file not found: %s (from file = %s)".formatted(filename,  Files.exists(soundPath)));
            }

            AudioInputStream audioIn = AudioSystem.getAudioInputStream(
                    new BufferedInputStream(audioInStream));
            Clip clip = AudioSystem.getClip();
            clip.open(audioIn);
            this.clip = clip;
        } catch(Exception e) {
            throw new FatalIOException("Failed to load sound: " + filename, e);
        }
    }

    public void export(Path exportPath) throws IOException{
        Path exportedFilePath = exportPath.resolve(filename);
        if (!Files.exists(exportedFilePath)) { // Check if the file already exists
            InputStream is = ClassLoader.getSystemClassLoader().getResourceAsStream("sounds/" + filename);
            if (is != null) {
                Files.copy(is, exportedFilePath, StandardCopyOption.REPLACE_EXISTING);
                is.close();
            }
        }
    }

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
