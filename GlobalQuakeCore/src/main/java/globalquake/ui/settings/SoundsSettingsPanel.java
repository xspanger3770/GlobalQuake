package globalquake.ui.settings;

import globalquake.core.GlobalQuake;
import globalquake.core.Settings;
import globalquake.core.exception.FatalIOException;
import globalquake.core.exception.RuntimeApplicationException;
import globalquake.sounds.GQSound;
import globalquake.sounds.Sounds;
import org.tinylog.Logger;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.io.IOException;

public class SoundsSettingsPanel extends SettingsPanel {

    private JSlider sliderMasterVolume;
    private JCheckBox chkBoxEnableSounds;

    public SoundsSettingsPanel() {
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        add(createMasterVolumeSlider());
        add(createIndividualSoundsPanel());
    }

    private Component createIndividualSoundsPanel() {
        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
        for (GQSound gqSound : Sounds.ALL_ACTUAL_SOUNDS) {
            var borderLayout = new BorderLayout();
            borderLayout.setHgap(10);
            borderLayout.setVgap(4);

            JPanel rootPanel = new JPanel(new BorderLayout());
            rootPanel.setBorder(BorderFactory.createRaisedBevelBorder());

            JPanel soundPanel = new JPanel(borderLayout);
            //soundPanel.setLayout(new BoxLayout(soundPanel, BoxLayout.X_AXIS));
            soundPanel.setSize(300, 300);

            JLabel label = new JLabel(gqSound.getFilename());
            label.setPreferredSize(new Dimension(160, 40));

            soundPanel.add(label, BorderLayout.WEST);

            JPanel volumePanel = new JPanel(new BorderLayout());
            JLabel labelVolume;
            volumePanel.add(labelVolume = new JLabel("Volume: %d%%".formatted((int) (gqSound.volume * 100.0))), BorderLayout.NORTH);

            JSlider volumeSlider = createSingleSoundVolumeSlider(gqSound, labelVolume);

            volumePanel.add(volumeSlider, BorderLayout.CENTER);

            soundPanel.add(volumePanel, BorderLayout.CENTER);

            JButton testSoundButton = new JButton("Test");
            testSoundButton.addActionListener(new AbstractAction() {
                @Override
                public void actionPerformed(ActionEvent actionEvent) {
                    Sounds.playSound(gqSound);
                }
            });

            JButton reloadSoundButton = new JButton("Reload");
            reloadSoundButton.addActionListener(new AbstractAction() {
                @Override
                public void actionPerformed(ActionEvent actionEvent) {
                    try {
                        gqSound.load(true);
                    } catch (FatalIOException e) {
                        if (GlobalQuake.getErrorHandler() != null) {
                            GlobalQuake.getErrorHandler().handleWarning(new RuntimeApplicationException("Failed to load this sound!", e));
                        } else {
                            Logger.error(e);
                        }
                    }
                }
            });

            var gl = new GridLayout(2, 1);
            gl.setHgap(2);
            gl.setVgap(4);

            JPanel p = new JPanel(gl);
            p.add(testSoundButton);
            p.add(reloadSoundButton);

            soundPanel.add(p, BorderLayout.EAST);

            rootPanel.add(soundPanel, BorderLayout.CENTER);

            JTextArea textAreaDescription = new JTextArea(gqSound.getDescription());
            textAreaDescription.setBorder(new EmptyBorder(5, 5, 5, 5));
            textAreaDescription.setEditable(false);
            textAreaDescription.setBackground(panel.getBackground());
            rootPanel.add(textAreaDescription, BorderLayout.SOUTH);

            panel.add(rootPanel);
        }

        return panel;
    }

    private static JSlider createSingleSoundVolumeSlider(GQSound gqSound, JLabel label) {
        JSlider volumeSlider = new JSlider(SwingConstants.HORIZONTAL, 0, 100, (int) (gqSound.volume * 100.0));
        volumeSlider.setMajorTickSpacing(10);
        volumeSlider.setMinorTickSpacing(5);
        volumeSlider.setPaintTicks(true);
        volumeSlider.setPaintLabels(true);

        volumeSlider.addChangeListener(changeEvent -> {
            gqSound.volume = volumeSlider.getValue() / 100.0;
            if (gqSound.equals(Sounds.countdown)) {
                Sounds.countdown2.volume = Sounds.countdown.volume; // workaround
            }
            label.setText("Volume: %d%%".formatted((int) (gqSound.volume * 100.0)));
        });
        return volumeSlider;
    }

    private Component createMasterVolumeSlider() {
        sliderMasterVolume = HypocenterAnalysisSettingsPanel.createSettingsSlider(0, 100, 10, 2);

        JLabel label = new JLabel();
        ChangeListener changeListener = changeEvent -> {
            label.setText("Master Volume: %d%%".formatted(
                    sliderMasterVolume.getValue()));
            Settings.globalVolume = sliderMasterVolume.getValue();
        };

        sliderMasterVolume.addChangeListener(changeListener);

        sliderMasterVolume.setValue(Settings.globalVolume);
        changeListener.stateChanged(null);


        JPanel coolLayout = HypocenterAnalysisSettingsPanel.createCoolLayout(sliderMasterVolume, label, null,
                null);

        JPanel fill1 = new JPanel();

        chkBoxEnableSounds = new JCheckBox("Enable sounds");
        chkBoxEnableSounds.setSelected(Settings.enableSound);
        fill1.add(chkBoxEnableSounds);

        chkBoxEnableSounds.addChangeListener(changeEvent -> Settings.enableSound = chkBoxEnableSounds.isSelected());

        JButton btnSoundsFolder = new JButton("Open Sounds Folder");
        btnSoundsFolder.addActionListener(new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                try {
                    Desktop.getDesktop().open(Sounds.EXPORT_DIR);
                } catch (IOException e) {
                    if (GlobalQuake.getErrorHandler() != null) {
                        GlobalQuake.getErrorHandler().handleWarning(new RuntimeApplicationException("Unable to open file explorer!", e));
                    } else {
                        Logger.error(e);
                    }
                }
            }
        });
        fill1.add(btnSoundsFolder);

        JButton btnReloadSounds = new JButton("Reload Sounds");
        btnReloadSounds.addActionListener(new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                try {
                    Sounds.loadSounds();
                } catch (Exception e) {
                    if (GlobalQuake.getErrorHandler() != null) {
                        GlobalQuake.getErrorHandler().handleWarning(new RuntimeApplicationException("Unable to reload sounds!", e));
                    } else {
                        Logger.error(e);
                    }
                }
            }
        });
        fill1.add(btnReloadSounds);
        coolLayout.add(fill1);

        return coolLayout;
    }

    @Override
    public void save() throws NumberFormatException {
        Sounds.storeVolumes();
    }

    @Override
    public String getTitle() {
        return "Sounds";
    }
}
