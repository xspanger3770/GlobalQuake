package globalquake.ui.globalquake;

import globalquake.main.Main;
import globalquake.ui.settings.Settings;
import globalquake.ui.settings.SettingsFrame;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class GlobalQuakeFrame extends JFrame {

	private boolean hideList = false;
	private final EarthquakeListPanel list;
	private final GlobalQuakePanel panel;
	private final JPanel mainPanel;
	private boolean _containsListToggle;

	public GlobalQuakeFrame() {
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		panel = new GlobalQuakePanel(this) {

			@Override
			public void paint(Graphics gr) {
				super.paint(gr);
				Graphics2D g = (Graphics2D) gr;
				g.setColor(_containsListToggle ? Color.gray : Color.lightGray);
				g.fillRect(getWidth() - 20, 0, 20, 30);
				g.setColor(Color.black);
				g.drawRect(getWidth() - 20, 0, 20, 30);
				g.setFont(new Font("Calibri", Font.BOLD, 16));
				g.setColor(Color.black);
				g.drawString(hideList ? "<" : ">", getWidth() - 16, 20);
			}
		};
		panel.addMouseListener(new MouseAdapter() {
			@Override
			public void mouseClicked(MouseEvent e) {
				int x = e.getX();
				int y = e.getY();
				if (x >= panel.getWidth() - 20 && x <= panel.getWidth() && y >= 0 && y <= 30) {
					toggleList();
				}
			}

			@Override
			public void mouseExited(MouseEvent e) {
				_containsListToggle = false;
			}
		});
		panel.addMouseMotionListener(new MouseAdapter() {

			@Override
			public void mouseMoved(MouseEvent e) {
				int x = e.getX();
				int y = e.getY();
				_containsListToggle = x >= panel.getWidth() - 20 && x <= panel.getWidth() && y >= 0 && y <= 30;
			}
		});

		list = new EarthquakeListPanel();
		panel.setPreferredSize(new Dimension(600, 600));
		list.setPreferredSize(new Dimension(300, 600));

		mainPanel = new JPanel();
		mainPanel.setLayout(new BorderLayout());
		mainPanel.setPreferredSize(new Dimension(1000, 700));
		mainPanel.add(panel, BorderLayout.CENTER);
		mainPanel.add(list, BorderLayout.EAST);

		setContentPane(mainPanel);

		setJMenuBar(createJMenuBar());

		pack();
		setLocationRelativeTo(null);
		setMinimumSize(new Dimension(610, 500));
		setResizable(true);
		setTitle(Main.fullName);

		ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);

		// Schedule the task
		scheduler.schedule(new Runnable() {
			@Override
			public void run() {
				mainPanel.repaint();
				scheduler.schedule(this, 1000 / Settings.fpsIdle, TimeUnit.MILLISECONDS);
			}
		}, 1, TimeUnit.SECONDS);
	}

	private JMenuBar createJMenuBar() {
		JMenuBar menuBar = new JMenuBar();
		menuBar.setBackground(Color.lightGray);

		JMenu menuOptions = new JMenu("Options");

		JMenuItem settings = new JMenuItem("Settings");
		settings.addActionListener(new AbstractAction() {
			@Override
			public void actionPerformed(ActionEvent actionEvent) {
				new SettingsFrame(GlobalQuakeFrame.this).setVisible(true);
			}
		});

		menuOptions.add(settings);

		menuBar.add(menuOptions);

		return menuBar;
	}

	protected void toggleList() {
		hideList = !hideList;
		if (hideList) {
			panel.setSize(new Dimension(mainPanel.getWidth(), mainPanel.getHeight()));
			list.setPreferredSize(new Dimension(0, (int) list.getPreferredSize().getHeight()));
		} else {
			panel.setSize(new Dimension(mainPanel.getWidth() - 300, mainPanel.getHeight()));
			list.setPreferredSize(new Dimension(300, (int) list.getPreferredSize().getHeight()));
		}
		_containsListToggle = false;
		revalidate();
	}

}
