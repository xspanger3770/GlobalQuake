package globalquake.ui.archived;

import globalquake.core.Settings;
import globalquake.core.archive.ArchivedQuake;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.time.Instant;

public class ArchivedQuakeUI extends JDialog {

    public ArchivedQuakeUI(Frame parent, ArchivedQuake quake) {
        super(parent);
        setLayout(new BorderLayout());

        JLabel latLabel = new JLabel("Latitude: %.4f".formatted(quake.getLat()));
        JLabel lonLabel = new JLabel("Longitude: %.4f".formatted(quake.getLon()));
        JLabel depthLabel = new JLabel("Depth: %s".formatted(Settings.getSelectedDistanceUnit().format(quake.getDepth(), 1)));
        JLabel originLabel = new JLabel("Origin Time: %s".formatted(Settings.formatDateTime(Instant.ofEpochMilli(quake.getOrigin()))));
        JLabel magLabel = new JLabel("Magnitude: %.2f".formatted(quake.getMag()));
        JLabel maxRatioLabel = new JLabel("Max Ratio: %.1f".formatted(quake.getMaxRatio()));
        JLabel regionLabel = new JLabel("Region: %s".formatted(quake.getRegion()));

        // Create a panel to hold the labels
        JPanel panel = new JPanel();
        panel.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
        panel.add(latLabel);
        panel.add(lonLabel);
        panel.add(depthLabel);
        panel.add(originLabel);
        panel.add(magLabel);
        panel.add(maxRatioLabel);
        panel.add(regionLabel);


        JButton animButton = new JButton("Animation");

        animButton.addActionListener(new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                new ArchivedQuakeAnimation(parent, quake).setVisible(true);
            }
        });

        getContentPane().add(panel, BorderLayout.CENTER);

        JPanel panel2 = new JPanel();
        panel2.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
        panel2.add(animButton);

        getContentPane().add(panel2, BorderLayout.SOUTH);

        for (Component component : panel.getComponents()) {
            component.setFont(new Font("Calibri", Font.PLAIN, 18));
        }

        addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                if (e.getKeyCode() == KeyEvent.VK_ESCAPE) {
                    dispose();
                }
            }
        });

        setTitle("M%.1f %s".formatted(quake.getMag(), quake.getRegion()));
        pack();
        setLocationRelativeTo(parent);
        setResizable(false);
    }
}
