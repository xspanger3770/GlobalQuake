package globalquake.ui.settings;

import globalquake.geo.taup.TauPTravelTimeCalculator;
import globalquake.ui.GQFrame;
import org.tinylog.Logger;

import javax.swing.*;
import java.awt.*;
import java.util.LinkedList;
import java.util.List;

import globalquake.main.Main;

public class SettingsFrame extends GQFrame {

	private final List<SettingsPanel> panels = new LinkedList<>();
	private JTabbedPane tabbedPane;

	public static void main(String[] args) throws Exception{
		TauPTravelTimeCalculator.init();
		EventQueue.invokeLater(() -> {
            try {
                new SettingsFrame(null).setVisible(true);
            } catch (Exception e) {
				Logger.error(e);
            }
        });
	}

	public SettingsFrame(Component parent) {
		initialize(parent);
	}

	private void initialize(Component parent) {
		setIconImage(Main.LOGO);

		setTitle("GlobalQuake Settings");
		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);

		JPanel contentPanel = new JPanel(new BorderLayout());
		setContentPane(contentPanel);

		tabbedPane = new JTabbedPane(JTabbedPane.TOP);
		contentPanel.add(tabbedPane, BorderLayout.CENTER);

		JPanel panel_1 = new JPanel();
		contentPanel.add(panel_1, BorderLayout.SOUTH);

		JButton btnCancel = new JButton("Cancel");
		panel_1.add(btnCancel);

		btnCancel.addActionListener(e -> dispose());

		JButton btnSave = new JButton("Save");
		panel_1.add(btnSave);

		btnSave.addActionListener(e -> {
            for (SettingsPanel panel1 : panels) {
                try {
                    panel1.save();
                } catch (Exception ex) {
                    error(ex);
                    return;
                }
            }
            Settings.save();
			dispose();
        });

		addPanels();

		pack();
		setResizable(false);
		setLocationRelativeTo(parent);
	}

	protected void error(Exception e) {
		JOptionPane.showMessageDialog(this, e.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
	}

	private void addPanels() {
		panels.add(new GeneralSettingsPanel(this));
		panels.add(new AlertSettingsPanel());
		panels.add(new PerformanceSettingsPanel());
		panels.add(new GraphicsSettingsPanel());
		panels.add(new CinemaModeSettingsPanel());
		panels.add(new HypocenterAnalysisSettingsPanel());
		panels.add(new DebugSettingsPanel());

		for (SettingsPanel panel : panels) {
			JScrollPane scrollPane = new JScrollPane(panel);
			scrollPane.setPreferredSize(new Dimension(700, 500));
			scrollPane.setHorizontalScrollBarPolicy(ScrollPaneConstants.HORIZONTAL_SCROLLBAR_NEVER);
			scrollPane.getVerticalScrollBar().setUnitIncrement(10);
			tabbedPane.addTab(panel.getTitle(), scrollPane);

			javax.swing.SwingUtilities.invokeLater(() -> scrollPane.getVerticalScrollBar().setValue(0));
		}
	}

	public void refreshUI() {
		for (SettingsPanel panel : panels) {
			panel.refreshUI();
		}
	}
}
