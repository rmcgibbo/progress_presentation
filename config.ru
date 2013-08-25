use Rack::Static, 
  :urls => ["/figures", "/videos", "/theme", "/js", "/css"],
  :root => "public"

use Rack::Auth::Basic, "Restricted Area" do |username, password|
  [username, password] == ['phd', 'phd']
end

run lambda { |env|
  [
    200, 
    {
      'Content-Type'  => 'text/html', 
      'Cache-Control' => 'public, max-age=86400' 
    },
    File.open('public/index.html', File::RDONLY)
  ]
}