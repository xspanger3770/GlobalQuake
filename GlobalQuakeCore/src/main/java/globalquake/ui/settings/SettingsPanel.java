package globalquake.ui.settings;


import globalquake.core.exception.RuntimeApplicationException;
import org.apache.commons.lang3.StringUtils;

import javax.swing.*;
import java.awt.*;

public abstract class SettingsPanel extends JPanel {
    private static final Insets WEST_INSETS = new Insets(5, 0, 5, 5);
    private static final Insets EAST_INSETS = new Insets(5, 5, 5, 0);

    public abstract void save() throws NumberFormatException;

    public abstract String getTitle();

    public void refreshUI() {
    }

    public double parseDouble(String str, String name, double min, double max) {
        double d = Double.parseDouble(str.replace(',', '.'));
        if (Double.isNaN(d) || Double.isInfinite(d)) {
            throw new RuntimeApplicationException("Invalid number: %s".formatted(str));
        }

        if (d < min || d > max) {
            throw new RuntimeApplicationException("%s must be between %s and %s (entered %s)!".formatted(name, min, max, d));
        }

        return d;
    }

    public int parseInt(String str, String name, int min, int max) {
        int n = Integer.parseInt(str);

        if (n < min || n > max) {
            throw new RuntimeApplicationException("%s must be between %s and %s (entered %s)!".formatted(name, min, max, n));
        }

        return n;
    }

    protected static GridBagConstraints createGbc(int y) {
        return createGbc(0, y, 2);
    }

    protected static GridBagConstraints createGbc(int x, int y) {
        return createGbc(x, y, 1);
    }

    protected static GridBagConstraints createGbc(int x, int y, int gridwidth) {
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.gridx = x;
        gbc.gridy = y;
        gbc.gridwidth = gridwidth;
        gbc.gridheight = 1;

        gbc.anchor = (x == 0) ? GridBagConstraints.WEST : GridBagConstraints.EAST;
        gbc.fill = (x == 0) ? GridBagConstraints.BOTH
                : GridBagConstraints.HORIZONTAL;

        gbc.insets = (x == 0) ? WEST_INSETS : EAST_INSETS;
        gbc.weightx = (x == 0) ? 0.1 : 1.0;
        gbc.weighty = 1.0;
        return gbc;
    }

    protected static Component createJTextArea(String text, Component parent) {
        JTextArea jTextArea = new JTextArea(text);
        jTextArea.setLineWrap(true);
        jTextArea.setEditable(false);
        jTextArea.setBackground(parent.getBackground());

        return jTextArea;
    }

    protected static JPanel createGridBagPanel() {
        return createGridBagPanel(true);
    }

    protected static JPanel createGridBagPanel(boolean emptyBorder) {
        return createGridBagPanel(null, emptyBorder);
    }

    protected static JPanel createGridBagPanel(String borderText) {
        return createGridBagPanel(borderText, false);
    }

    protected static JPanel createGridBagPanel(String borderText, boolean emptyBorder) {
        JPanel panel = new JPanel(new GridBagLayout());
        if (StringUtils.isEmpty(borderText)) {
            if (emptyBorder) {
                panel.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
            }
        } else {
            panel.setBorder(BorderFactory.createTitledBorder(borderText));
        }
        return panel;
    }

    protected static JPanel createVerticalPanel() {
        return createVerticalPanel(true);
    }

    protected static JPanel createVerticalPanel(boolean emptyBorder) {
        return createVerticalPanel(null, emptyBorder);
    }

    protected static JPanel createVerticalPanel(String borderText) {
        return createVerticalPanel(borderText, false);
    }

    protected static JPanel createVerticalPanel(String borderText, boolean emptyBorder) {
        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
        if (StringUtils.isEmpty(borderText)) {
            if (emptyBorder) {
                panel.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
            }
        } else {
            panel.setBorder(BorderFactory.createTitledBorder(borderText));
        }
        return  panel;
    }

    protected static JPanel createHorizontalPanel() {
        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.X_AXIS));
        return panel;
    }

    protected static Component alignLeft(Component component) {
        JPanel panel = new JPanel(new BorderLayout());
        panel.add(component, BorderLayout.WEST);
        return panel;
    }
}
