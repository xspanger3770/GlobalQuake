package globalquake.ui.settings;

import globalquake.core.intensity.IntensityScale;
import globalquake.core.intensity.IntensityScales;
import globalquake.core.intensity.Level;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.util.Objects;

public class IntensityScaleSelector extends JPanel {

    private final JComboBox<IntensityScale> shakingScaleComboBox;
    private final JComboBox<Level> levelComboBox;

    public IntensityScaleSelector(String text, int shakingLevelScale, int shakingLevelIndex) {
        shakingScaleComboBox = new JComboBox<>(IntensityScales.INTENSITY_SCALES);
        shakingScaleComboBox.setSelectedIndex(Math.max(0, Math.min(shakingScaleComboBox.getItemCount() - 1, shakingLevelScale)));
        levelComboBox = new JComboBox<>(((IntensityScale) Objects.requireNonNull(shakingScaleComboBox.getSelectedItem())).getLevels().toArray(new Level[0]));

        levelComboBox.setSelectedIndex(Math.max(0, Math.min(levelComboBox.getItemCount() - 1, shakingLevelIndex)));

        shakingScaleComboBox.addActionListener(new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                levelComboBox.removeAllItems();
                ((IntensityScale) shakingScaleComboBox.getSelectedItem()).getLevels().forEach(levelComboBox::addItem);
            }
        });

        add(new JLabel(text));
        add(shakingScaleComboBox);
        add(levelComboBox);
    }

    public JComboBox<Level> getLevelComboBox() {
        return levelComboBox;
    }

    public JComboBox<IntensityScale> getShakingScaleComboBox() {
        return shakingScaleComboBox;
    }
}
