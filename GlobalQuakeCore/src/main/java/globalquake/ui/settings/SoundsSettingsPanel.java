package globalquake.ui.settings;

import globalquake.core.GlobalQuake;
import globalquake.core.Settings;
import globalquake.core.exception.FatalIOException;
import globalquake.core.exception.RuntimeApplicationException;
import globalquake.sounds.GQSound;
import globalquake.sounds.Sounds;
import org.tinylog.Logger;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.io.IOException;

public class SoundsSettingsPanel extends SettingsPanel {
    private JSlider sliderMasterVolume;
    private JCheckBox chkBoxEnableSounds;

    public SoundsSettingsPanel() {
        setLayout(new BorderLayout());

        JPanel panel = createVerticalPanel(false);
        panel.add(createMasterVolumeSlider());
        panel.add(createIndividualSoundsPanel());

        add(panel, BorderLayout.NORTH);
    }

    private Component createMasterVolumeSlider() {
        sliderMasterVolume = HypocenterAnalysisSettingsPanel.createSettingsSlider(0, 100, 10, 2);

        JLabel label = new JLabel("Master Volume: %d%%".formatted(sliderMasterVolume.getValue()));
        sliderMasterVolume.addChangeListener(e -> label.setText("Master Volume: %d%%".formatted(sliderMasterVolume.getValue())));
        sliderMasterVolume.setValue(Settings.globalVolume);

        JPanel coolLayout = HypocenterAnalysisSettingsPanel.createCoolLayout(sliderMasterVolume, label, null, null);

        JPanel fill1 = new JPanel();
        fill1.add(chkBoxEnableSounds = new JCheckBox("Enable sounds.", Settings.enableSound));

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
        coolLayout.add(fill1, createGbc(4));

        return coolLayout;
    }

    private Component createIndividualSoundsPanel() {
        JPanel panel = createVerticalPanel();
        for (GQSound gqSound : Sounds.ALL_ACTUAL_SOUNDS) {
            JPanel rootPanel = new JPanel(new BorderLayout());
            rootPanel.setBorder(BorderFactory.createRaisedBevelBorder());

            var borderLayout = new BorderLayout();
            borderLayout.setHgap(10);
            borderLayout.setVgap(4);
            JPanel soundPanel = new JPanel(borderLayout);

            JLabel label = new JLabel(gqSound.getFilename());
            label.setPreferredSize(new Dimension(160, 40));

            soundPanel.add(label, BorderLayout.WEST);

            JPanel volumePanel = new JPanel(new BorderLayout());
            JLabel labelVolume = new JLabel("Volume: %d%%".formatted((int) (gqSound.volume * 100.0)));
            volumePanel.add(labelVolume, BorderLayout.NORTH);

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

            rootPanel.add(createJTextArea(gqSound.getDescription(), panel), BorderLayout.SOUTH);

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

    @Override
    public void save() throws NumberFormatException {
        Settings.globalVolume = sliderMasterVolume.getValue();
        Settings.enableSound = chkBoxEnableSounds.isSelected();
        Sounds.storeVolumes();
    }

    @Override
    public String getTitle() {
        return "Sounds";
    }
}
